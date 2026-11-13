"""PatchCore 在 MVTecAD pill 上的训练 + 测试评估。

输出:
- Lightning 自动记录的 image/pixel AUROC、F1Score(控制台 + results/ CSV)
- 从测试批次聚合出的混淆矩阵指标:accuracy / precision / recall / F1

注意:默认 ``val_split_mode=SAME_AS_TEST``,自适应阈值是在测试集上拟合的,
因此阈值相关指标(F1/precision/recall/accuracy)会偏乐观。
更严格的评估可在 MVTecAD(...) 中加:
    val_split_mode=ValSplitMode.SYNTHETIC, val_split_ratio=0.1
"""

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

if __name__ == "__main__":
    datamodule = MVTecAD(root=None, category="pill")

    model = Patchcore(
        coreset_sampling_ratio=0.01,  # 参数扫描验证:pill 上比默认 0.1 显著更优(AUROC 0.94 → 0.96+)
    )

    engine = Engine(max_epochs=1)

    # 1. 训练(构建 memory bank + 拟合阈值)
    engine.fit(model=model, datamodule=datamodule)

    # 2. 测试:返回的 results 是 [指标字典],包含 image_AUROC 等四个汇总指标
    #    (控制台/CSV 同样会输出)
    results = engine.test(model=model, datamodule=datamodule)
    if results:
        print("\n===== Lightning 记录的测试指标 =====")
        for name, value in results[0].items():
            value = value.item() if hasattr(value, "item") else value
            print(f"{name}: {value:.4f}")

    # 3. 对测试集逐图预测,聚合混淆矩阵指标
    #    predict 默认走 test_dataloader,输出含 pred_score/pred_label/gt_label
    predictions = engine.predict(model=model, datamodule=datamodule)
    scores = torch.cat([p["pred_score"].cpu() for p in predictions])
    labels = torch.cat([p["gt_label"].cpu() for p in predictions])
    preds = torch.cat([p["pred_label"].cpu() for p in predictions])

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    print("\n===== 图像级混淆矩阵 =====")
    print(f"            预测正常    预测异常")
    print(f"真实正常    {tn:>6}    {fp:>6}")
    print(f"真实异常    {fn:>6}    {tp:>6}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("\n===== 图像级指标(自适应阈值下) =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # 4. 可视化 demo:用 anomalib 内置的 ImageVisualizer 把 heatmap 叠加到原图并保存
    from pathlib import Path

    from anomalib.visualization import ImageVisualizer

    # 只叠加 anomaly_map 热力图(默认还会叠加 pred_mask 红色掩码)
    visualizer = ImageVisualizer(
        fields=["image"],
        overlay_fields=[("image", ["anomaly_map"])],
    )
    out_dir = Path("results/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {"normal": 0, "anomalous": 0}
    max_per_class = 4
    for p in predictions:
        vis_images = visualizer.visualize(p)  # 每张图一个 PIL.Image(或 None)
        labels_b = p["gt_label"]
        for i, vis in enumerate(vis_images):
            if vis is None:
                continue
            cls = "anomalous" if labels_b[i] == 1 else "normal"
            if saved[cls] >= max_per_class:
                continue
            saved[cls] += 1
            name = Path(p["image_path"][i]).stem
            vis.save(out_dir / f"{cls}_{name}_heatmap.png")
        if saved["normal"] >= max_per_class and saved["anomalous"] >= max_per_class:
            break

    print(
        f"\n===== 可视化 demo:已保存 {saved['normal']} 张正常图 + {saved['anomalous']} 张异常图的热力图到 {out_dir.resolve()} ====="
    )
