"""PatchCore 在 pill 上的参数扫描。

对比 backbone / layers / coreset_sampling_ratio / num_neighbors 对
image_AUROC、F1、accuracy、precision、recall 的影响。
固定 seed=42 以保证可比性。输出行带 "SWEEP:" 前缀,便于过滤。
"""

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

CONFIGS = [
    {"name": "baseline: wr50_2, L2+L3, ratio=0.10, nn=9", "kwargs": {}},
    {"name": "num_neighbors=25", "kwargs": {"num_neighbors": 25}},
    {"name": "coreset_ratio=0.25", "kwargs": {"coreset_sampling_ratio": 0.25}},
    {"name": "layers: L1+L2+L3", "kwargs": {"layers": ("layer1", "layer2", "layer3")}},
    {"name": "backbone=resnet50", "kwargs": {"backbone": "resnet50"}},
    {"name": "coreset_ratio=0.01", "kwargs": {"coreset_sampling_ratio": 0.01}},
]


def evaluate(kwargs: dict) -> dict:
    """跑一轮 fit + test + predict,返回全部指标。"""
    torch.manual_seed(42)
    datamodule = MVTecAD(root=None, category="pill", seed=42)
    model = Patchcore(**kwargs)
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
    import sys

    # 可选:命令行传入要跑的配置下标,如 `python patchcore_pill_sweep.py 4 5`
    only = {int(a) for a in sys.argv[1:]} if len(sys.argv) > 1 else None

    results = []
    for i, cfg in enumerate(CONFIGS):
        if only is not None and i not in only:
            continue
        r = evaluate(cfg["kwargs"])
        results.append((cfg["name"], r))
        print(
            f"SWEEP: {cfg['name']:40s} | AUROC={r['image_AUROC']:.4f} "
            f"F1={r['image_F1Score']:.4f} ACC={r['accuracy']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f} "
            f"[TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}]",
            flush=True,
        )

    print("\n===== SWEEP SUMMARY =====")
    for name, r in results:
        print(
            f"SWEEP: {name:40s} AUROC={r['image_AUROC']:.4f} "
            f"F1={r['image_F1Score']:.4f} ACC={r['accuracy']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f}"
        )
