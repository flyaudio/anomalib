"""跨类别区分实验:pill vs screw,用 yolo11m 骨干(对照 wide_resnet50_2)。

两个方向:
  A. 用 pill 训练 PatchCore → 测试区分 pill(正常)vs screw(异常)
  B. 用 screw 训练 PatchCore → 测试区分 screw(正常)vs pill(异常)

评测集构造(Folder 格式):
  good/       = 正常类别的全部训练图
  test/good/  = 正常类别的测试正常图
  bad/        = 异常类别的全部图(测试时全部视为"异常")
"""

import shutil
from pathlib import Path

import torch
from torch import nn

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore
from ultralytics import YOLO

from yolo_backbone_patchcore import YoloBackbone, patch_c2f_for_fx

BASE = Path.home() / ".cache/anomalib/datasets/MVTecAD"
SEED = 42


def copy_resized(src: Path, dst: Path, size: int = 256) -> None:
    """复制并缩放到 size×size(MVTecAD 原图 800/1024,不缩放 patch 数会爆显存)。"""
    from PIL import Image

    Image.open(src).convert("RGB").resize((size, size)).save(dst)


def build_cross_dataset(normal_cat: str, anomalous_cat: str) -> Path:
    """构造 cross_{normal_cat} 评测集:good=normal 训练图,bad=anomalous 全部图。"""
    root = Path(f"datasets/cross_{normal_cat}")
    if root.exists():
        shutil.rmtree(root)
    for d in [root / "good", root / "bad", root / "test" / "good"]:
        d.mkdir(parents=True)

    for p in sorted((BASE / normal_cat / "train" / "good").glob("*.png")):
        copy_resized(p, root / "good" / p.name)
    for p in sorted((BASE / normal_cat / "test" / "good").glob("*.png")):
        copy_resized(p, root / "test" / "good" / p.name)

    # 异常类别的全部图(train good + test 各子目录),加前缀避免同名冲突
    anom = [(f"train_{p.name}", p) for p in sorted((BASE / anomalous_cat / "train" / "good").glob("*.png"))]
    for sub in sorted(d for d in (BASE / anomalous_cat / "test").iterdir() if d.is_dir()):
        anom += [(f"{sub.name}_{p.name}", p) for p in sorted(sub.glob("*.png"))]
    for name, p in anom:
        copy_resized(p, root / "bad" / name)

    n_good = len(list((root / "good").glob("*.png")))
    n_test_ok = len(list((root / "test" / "good").glob("*.png")))
    print(f"评测集 {root}:训练 OK {n_good},测试 OK {n_test_ok},测试异常(全部 {anomalous_cat} 图) {len(anom)}")
    return root


def build_yolo11m_backbone() -> YoloBackbone:
    """yolo11m 骨干(0-6 层,P4 结束)。"""
    patch_c2f_for_fx()
    yolo = YOLO("yolo11m.pt")
    return YoloBackbone(yolo.model.model[:7])


def evaluate_patchcore(model: Patchcore, eval_root: Path, tag: str) -> dict:
    """在评测集上 fit + test + predict。"""
    torch.manual_seed(SEED)
    datamodule = Folder(
        name=eval_root.name,
        root=eval_root,
        normal_dir="good",
        abnormal_dir="bad",
        normal_test_dir="test/good",
        normal_split_ratio=0.0,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
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
        f"RESULT: {tag:36s} | AUROC={out['image_AUROC']:.4f} F1={out['image_F1Score']:.4f} "
        f"ACC={out['accuracy']:.4f} P={out['precision']:.4f} R={out['recall']:.4f} "
        f"[TN={tn} FP={fp} FN={fn} TP={tp}]",
        flush=True,
    )
    return out


def main() -> None:
    results = []
    for normal_cat, anomalous_cat in [("pill", "screw"), ("screw", "pill")]:
        eval_root = build_cross_dataset(normal_cat, anomalous_cat)

        print(f"\n===== 方向:训练 {normal_cat} → 区分 {normal_cat}/{anomalous_cat} =====")

        print(f"\n--- yolo11m 骨干 ---")
        yolo_backbone = build_yolo11m_backbone()
        results.append(
            (
                f"yolo11m  train={normal_cat}",
                evaluate_patchcore(
                    Patchcore(
                        backbone=yolo_backbone,
                        layers=["seq.4", "seq.6"],
                        pre_processor=False,
                        coreset_sampling_ratio=0.01,
                    ),
                    eval_root,
                    f"yolo11m train={normal_cat}",
                ),
            )
        )

        print(f"\n--- wide_resnet50_2 骨干(参考) ---")
        results.append(
            (
                f"wr50_2   train={normal_cat}",
                evaluate_patchcore(
                    Patchcore(coreset_sampling_ratio=0.01),
                    eval_root,
                    f"wr50_2 train={normal_cat}",
                ),
            )
        )

    print("\n===== 跨类别区分最终对比 =====")
    for name, r in results:
        print(
            f"RESULT: {name:24s} | AUROC={r['image_AUROC']:.4f} F1={r['image_F1Score']:.4f} "
            f"ACC={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
            f"[TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}]"
        )


if __name__ == "__main__":
    main()
