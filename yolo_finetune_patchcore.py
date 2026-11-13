"""路线 1 实验:OK/NG 二分类微调 YOLO11n 骨干 → 接回 PatchCore,对比微调前后效果。

无泄漏设计:
- 微调(有监督):OK = train/good(209 张);NG = 4 种"已见"缺陷类型
  (color / combined / contamination / crack)
- PatchCore 评测:OK = test/good(26 张);NG = 3 种"未见过"缺陷类型
  (faulty_imprint / pill_type / scratch)——测微调骨干对新缺陷类型的泛化

对比(同评测集、同 seed、同 coreset=0.01):
  1. wide_resnet50_2(ImageNet 预训练)—— 上界参考
  2. yolo11n(COCO 预训练,未微调)
  3. yolo11n(OK/NG 二分类微调后)
"""

import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore
from ultralytics import YOLO

from yolo_backbone_patchcore import YoloBackbone, patch_c2f_for_fx

PILL = Path.home() / ".cache/anomalib/datasets/MVTecAD/pill"
SEEN_TYPES = ["color", "combined", "contamination", "crack"]  # 参与微调的缺陷类型
HELD_TYPES = ["faulty_imprint", "pill_type", "scratch"]  # 留出测试的缺陷类型
EVAL_ROOT = Path("datasets/finetune_eval")  # Folder 格式的评测集(仅含留出类型)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

TRAIN_TRANSFORM = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ]
)


# ---------------------------------------------------------------- 数据准备


class ImageDataset(Dataset):
    """简单图像分类数据集:OK=0, NG=1。图片在 [0,1] 的 float tensor。"""

    def __init__(self, paths: list[Path], labels: list[int], transform: v2.Transform | None = None) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import torchvision
        from torchvision.transforms.v2.functional import resize

        img = torchvision.io.read_image(str(self.paths[idx]), torchvision.io.ImageReadMode.RGB)
        img = self.to_tensor(img)
        img = resize(img, [256, 256], antialias=True)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def build_finetune_data() -> tuple[DataLoader, DataLoader]:
    """构造微调数据:OK + 已见 NG,按 80/20 切训练/验证(固定 seed)。"""
    gen = torch.Generator().manual_seed(SEED)
    pos = sorted((PILL / "train" / "good").glob("*.png"))
    neg = [f for t in SEEN_TYPES for f in sorted((PILL / "test" / t).glob("*.png"))]

    n_pos = len(pos)
    n_neg = len(neg)
    all_idx = torch.randperm(n_pos + n_neg, generator=gen)
    all_paths = pos + neg
    all_labels = [0] * n_pos + [1] * n_neg

    n_val = (n_pos + n_neg) // 5
    val_idx = set(all_idx[:n_val].tolist())
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    for i, idx in enumerate(all_idx.tolist()):
        (train_paths if i not in val_idx else val_paths).append(all_paths[idx])
        (train_labels if i not in val_idx else val_labels).append(all_labels[idx])

    train_ds = ImageDataset(train_paths, train_labels, TRAIN_TRANSFORM)
    val_ds = ImageDataset(val_paths, val_labels)
    print(f"微调数据:训练 {len(train_ds)}(OK {train_labels.count(0)}/NG {train_labels.count(1)}), 验证 {len(val_ds)}")
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4),
    )


def build_eval_folder() -> None:
    """构造 Folder 格式的评测集:训练=全部 OK 训练图,测试=OK + 留出 NG 类型。"""
    if EVAL_ROOT.exists():
        shutil.rmtree(EVAL_ROOT)
    for d in [EVAL_ROOT / "good", EVAL_ROOT / "bad", EVAL_ROOT / "test" / "good", EVAL_ROOT / "test" / "bad"]:
        d.mkdir(parents=True)

    for f in sorted((PILL / "train" / "good").glob("*.png")):
        shutil.copy(f, EVAL_ROOT / "good" / f.name)
    for f in sorted((PILL / "test" / "good").glob("*.png")):
        shutil.copy(f, EVAL_ROOT / "test" / "good" / f.name)
    held = [(t, f) for t in HELD_TYPES for f in sorted((PILL / "test" / t).glob("*.png"))]
    for t, f in held:
        # 各缺陷类型目录内文件名相同,加类型前缀避免冲突(也避免覆盖只读文件)
        name = f"{t}_{f.name}"
        shutil.copy(f, EVAL_ROOT / "test" / "bad" / name)
        shutil.copy(f, EVAL_ROOT / "bad" / name)
    print(f"评测集:训练 OK {len(list((EVAL_ROOT / 'good').glob('*.png')))}, "
          f"测试 OK {len(list((EVAL_ROOT / 'test' / 'good').glob('*.png')))}, "
          f"测试 NG {len(held)}(未见过类型 {HELD_TYPES})")


# ---------------------------------------------------------------- 骨干微调


class ClfHead(nn.Module):
    """GAP + FC 二分类头。"""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class ClfModel(nn.Module):
    """骨干 + 分类头。入口把 [0,1] 放大到 [0,255],与 PatchCore 使用时一致。"""

    def __init__(self, seq: nn.Module, in_ch: int) -> None:
        super().__init__()
        self.seq = seq
        self.head = ClfHead(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.seq(x * 255.0))


def finetune_backbone(seq: nn.Module, in_ch: int) -> None:
    """在 OK/NG 二分类任务上微调骨干(原地更新 seq 的权重)。"""
    # 关键:ultralytics 加载的权重 requires_grad=False,必须显式打开
    for p in seq.parameters():
        p.requires_grad = True
    train_loader, val_loader = build_finetune_data()
    model = ClfModel(seq, in_ch).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += len(y)
            correct += (model(x).argmax(1) == y).sum().item()
            running += loss.item() * len(y)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{EPOCHS}: loss {running / total:.4f}, train acc {correct / total:.4f}")

    # 验证集 + 留出类型上的分类准确率(仅 sanity check)
    model.eval()
    with torch.no_grad():
        val_total = val_correct = 0
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            val_total += len(y)
            val_correct += (model(x).argmax(1) == y).sum().item()
        print(f"分类器验证集准确率: {val_correct / val_total:.4f}")

        # 留出 NG 类型:分类器是否认得(预期会明显下降,这正是留出的意义)
        ok_imgs = [f for f in sorted((PILL / "test" / "good").glob("*.png"))]
        held_imgs = [f for t in HELD_TYPES for f in sorted((PILL / "test" / t).glob("*.png"))]
        held_ds = ImageDataset(ok_imgs + held_imgs, [0] * len(ok_imgs) + [1] * len(held_imgs))
        held_loader = DataLoader(held_ds, batch_size=BATCH_SIZE, num_workers=4)
        held_total = held_correct = 0
        for x, y in held_loader:
            x, y = x.cuda(), y.cuda()
            held_total += len(y)
            held_correct += (model(x).argmax(1) == y).sum().item()
        print(f"分类器在'未见过缺陷类型'上的准确率: {held_correct / held_total:.4f}")


# ---------------------------------------------------------------- PatchCore 评测


def evaluate_patchcore(model: Patchcore, tag: str) -> dict:
    """在 Folder 评测集上 fit + test + predict。"""
    torch.manual_seed(SEED)
    datamodule = Folder(
        name="finetune_eval",
        root=EVAL_ROOT,
        normal_dir="good",  # 训练 OK
        abnormal_dir="bad",  # 异常图来源是 root/bad(不是 test/bad!)
        normal_test_dir="test/good",  # 测试 OK
        normal_split_ratio=0.0,  # 267 张全部用于训练,不再从 good 切测试
        val_split_mode=ValSplitMode.SAME_AS_TEST,  # 与 MVTecAD 实验一致(阈值在测试集上拟合,三组可比)
        seed=SEED,
        num_workers=4,
    )
    engine = Engine(max_epochs=1)
    engine.fit(model=model, datamodule=datamodule)

    metrics = engine.test(model=model, datamodule=datamodule)[0]
    out = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}

    predictions = engine.predict(model=model, datamodule=datamodule)
    labels = torch.cat([p["gt_label"].cpu() for p in predictions])
    preds = torch.cat([p["pred_label"].cpu() for p in predictions])
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    out.update(
        {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    )
    print(
        f"RESULT: {tag:32s} | AUROC={out['image_AUROC']:.4f} F1={out['image_F1Score']:.4f} "
        f"ACC={out['accuracy']:.4f} P={out['precision']:.4f} R={out['recall']:.4f} "
        f"[TN={tn} FP={fp} FN={fn} TP={tp}]",
        flush=True,
    )
    return out


def load_coco_backbone() -> nn.Module:
    """加载 COCO 预训练 yolo11n 的纯顺序骨干(0-6 层,P4 结束)。"""
    yolo = YOLO("yolo11n.pt")
    return yolo.model.model[:7]


if __name__ == "__main__":
    build_eval_folder()
    patch_c2f_for_fx()  # FX 补丁,进程内只需一次

    # 1. 微调:COCO 骨干 → OK/NG 二分类微调(原地更新权重)
    print("\n===== 步骤 1:OK/NG 二分类微调 yolo11n 骨干 =====")
    seq = load_coco_backbone()
    finetune_backbone(seq, in_ch=128)  # yolo11n P4 = 128ch

    # 2. 三组 PatchCore 对比(同一评测集)
    print("\n===== 步骤 2:PatchCore 对比(评测集 = OK + 未见过缺陷类型) =====")

    results = {}

    print("\n--- 2a. wide_resnet50_2 (ImageNet) ---")
    results["wide_resnet50_2"] = evaluate_patchcore(Patchcore(coreset_sampling_ratio=0.01), "wide_resnet50_2")

    print("\n--- 2b. yolo11n (COCO 未微调) ---")
    coco_backbone = YoloBackbone(load_coco_backbone())
    results["yolo11n_coco"] = evaluate_patchcore(
        Patchcore(
            backbone=coco_backbone,
            layers=["seq.4", "seq.6"],
            pre_processor=False,
            coreset_sampling_ratio=0.01,
        ),
        "yolo11n_coco",
    )

    print("\n--- 2c. yolo11n (OK/NG 微调后) ---")
    finetuned_backbone = YoloBackbone(seq)
    results["yolo11n_finetuned"] = evaluate_patchcore(
        Patchcore(
            backbone=finetuned_backbone,
            layers=["seq.4", "seq.6"],
            pre_processor=False,
            coreset_sampling_ratio=0.01,
        ),
        "yolo11n_finetuned",
    )

    print("\n===== 最终对比(评测集:OK + 未见过缺陷类型) =====")
    for name, r in results.items():
        print(
            f"RESULT: {name:20s} | AUROC={r['image_AUROC']:.4f} F1={r['image_F1Score']:.4f} "
            f"ACC={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
            f"[TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}]"
        )
