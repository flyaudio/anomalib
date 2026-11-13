"""验证 ratio=0.01 的多 seed 稳健性,以及最优参数组合。"""

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

CONFIGS = [
    {"name": "ratio=0.01, seed=42", "kwargs": {"coreset_sampling_ratio": 0.01}, "seed": 42},
    {"name": "ratio=0.01, seed=123", "kwargs": {"coreset_sampling_ratio": 0.01}, "seed": 123},
    {"name": "ratio=0.01, seed=777", "kwargs": {"coreset_sampling_ratio": 0.01}, "seed": 777},
    {
        "name": "ratio=0.01 + nn=25 + L1L2L3",
        "kwargs": {
            "coreset_sampling_ratio": 0.01,
            "num_neighbors": 25,
            "layers": ("layer1", "layer2", "layer3"),
        },
        "seed": 42,
    },
]


def evaluate(kwargs: dict, seed: int) -> dict:
    """跑一轮 fit + test + predict,返回全部指标。"""
    torch.manual_seed(seed)
    datamodule = MVTecAD(root=None, category="pill", seed=seed)
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
    results = []
    for cfg in CONFIGS:
        r = evaluate(cfg["kwargs"], cfg["seed"])
        results.append((cfg["name"], r))
        print(
            f"SWEEP: {cfg['name']:32s} | AUROC={r['image_AUROC']:.4f} "
            f"F1={r['image_F1Score']:.4f} ACC={r['accuracy']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f} "
            f"[TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}]",
            flush=True,
        )

    print("\n===== VERIFY SUMMARY =====")
    for name, r in results:
        print(
            f"SWEEP: {name:32s} AUROC={r['image_AUROC']:.4f} "
            f"F1={r['image_F1Score']:.4f} ACC={r['accuracy']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f}"
        )
