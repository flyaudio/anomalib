"""实验:YOLO11 骨干(COCO 预训练) vs wide_resnet50_2(ImageNet 预训练)在 pill 上的 PatchCore 效果。

- YOLO 骨干:切出 backbone 纯顺序部分,提取 P3/P4 特征图
- 关闭 anomalib 默认 ImageNet 归一化(pre_processor=False),在 backbone 入口把
  [0,1] 输入放大到 [0,255](YOLO 训练时的输入范围)
- 所有配置 coreset_sampling_ratio=0.01, seed=42
"""

import torch
from torch import nn

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from ultralytics import YOLO

VARIANTS = [
    {
        "name": "wide_resnet50_2 (ImageNet)",
        "urls": None,
    },
    {
        "name": "yolo11n P3+P4 (COCO)",
        "urls": [
            "yolo11n.pt",
            "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
        ],
    },
    {
        "name": "yolo11m P3+P4 (COCO)",
        "urls": [
            "yolo11m.pt",
            "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt",
        ],
    },
]


class YoloBackbone(nn.Module):
    """把 YOLO 的 backbone Sequential 包装成可 FX trace 的模块,入口放大输入。"""

    def __init__(self, seq: nn.Module) -> None:
        super().__init__()
        self.seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x * 255.0)


def patch_c2f_for_fx() -> None:
    """把 C2f.forward 换成数值等价的 FX 可追踪版本(C3k2 继承 C2f)。

    原实现用 ``list(self.cv1(x).chunk(2, 1))`` + extend,FX 无法追踪 proxy 的
    列表迭代;改为 Python list 追加(数值完全等价:C2f 会把每个 block 的输出
    都拼起来)。
    """
    from ultralytics.nn.modules.block import C2f

    def fx_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).chunk(2, 1)
        pieces = [y[0], y[1]]
        for m in self.m:
            pieces.append(m(pieces[-1]))
        return self.cv2(torch.cat(pieces, 1))

    C2f.forward = fx_forward


def load_yolo11(urls: list[str]) -> tuple[nn.Module, dict[int, int]]:
    """加载 YOLO11,打印 backbone 各层输出形状,返回 (Sequential, {层索引: 通道数})。"""
    yolo = None
    for url in urls:
        try:
            yolo = YOLO(url)
            break
        except Exception as e:  # noqa: BLE001
            print(f"下载失败 {url}: {e}")
    if yolo is None:
        raise RuntimeError("YOLO 权重下载失败")

    layers = yolo.model.model  # Sequential
    x = torch.zeros(1, 3, 256, 256)
    shapes = {}
    with torch.no_grad():
        # 只逐层跑纯顺序的 backbone 部分(索引 11 起是带跳跃连接的 neck)
        for i, layer in enumerate(layers[:11]):
            x = layer(x)
            if isinstance(x, torch.Tensor):
                shapes[i] = tuple(x.shape)
                print(f"  YOLO_LAYER: {i:3d} {type(layer).__name__:10s} -> {tuple(x.shape)}")
    return layers, shapes


def build_model(variant: dict) -> Patchcore:
    """根据配置构建 Patchcore 模型。"""
    if variant["urls"] is None:
        # 基线:ImageNet 预训练的 wide_resnet50_2
        return Patchcore(coreset_sampling_ratio=0.01)

    layers, shapes = load_yolo11(variant["urls"])
    # 自动找 P3/P4:纯顺序部分里最后一个 32×32 和 16×16 输出
    p3_idx = max(i for i, s in shapes.items() if s[2] == 32)
    p4_idx = max(i for i, s in shapes.items() if s[2] == 16)
    print(
        f"  使用层: 索引 {p3_idx}(P3, {shapes[p3_idx][1]}ch) + {p4_idx}(P4, {shapes[p4_idx][1]}ch)"
    )

    patch_c2f_for_fx()  # 必须在创建 FX extractor 之前打补丁
    backbone = YoloBackbone(layers[: p4_idx + 1])
    return Patchcore(
        backbone=backbone,
        layers=[f"seq.{p3_idx}", f"seq.{p4_idx}"],
        pre_processor=False,  # 关闭 ImageNet 归一化,由 backbone 入口放大到 [0,255]
        coreset_sampling_ratio=0.01,
    )


def evaluate(model: Patchcore) -> dict:
    """fit + test + predict,返回全部指标。"""
    torch.manual_seed(42)
    datamodule = MVTecAD(root=None, category="pill", seed=42)
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
    return out


if __name__ == "__main__":
    results = []
    for variant in VARIANTS:
        print(f"\n===== 实验: {variant['name']} =====")
        model = build_model(variant)
        r = evaluate(model)
        results.append((variant["name"], r))
        print(
            f"COMPARE: {variant['name']:32s} | AUROC={r['image_AUROC']:.4f} "
            f"F1={r['image_F1Score']:.4f} ACC={r['accuracy']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f} "
            f"[TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}]",
            flush=True,
        )

    print("\n===== 最终对比 =====")
    for name, r in results:
        print(
            f"COMPARE: {name:32s} AUROC={r['image_AUROC']:.4f} "
            f"F1={r['image_F1Score']:.4f} ACC={r['accuracy']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f} "
            f"[TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}]"
        )
