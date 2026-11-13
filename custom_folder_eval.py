"""自建数据集(Folder)评估模板:训练 + 测试 + 混淆矩阵指标。

目录结构(名称可在下方常量里改,已对照 Folder 源码验证):

    datasets/my_dataset/
    ├── good/                      # 正常样本(训练)
    ├── bad/                       # 异常样本(测试)——注意:测试异常图来自这里!
    ├── test/
    │   └── good/                  # 正常样本(测试,经 normal_test_dir 指定)
    └── ground_truth/              # 可选:异常掩码,用于像素级指标
        └── bad/                   # 与 bad/ 同名,后缀 _mask.png

用法:
    1. 把数据按上面结构放好
    2. 修改下方 DATASET_ROOT / NORMAL_DIR / ABNORMAL_DIR
    3. python custom_folder_eval.py
"""

import torch

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

DATASET_ROOT = "./datasets/my_dataset"  # ← 改成你的数据路径
NORMAL_DIR = "good"  # 正常样本目录名
ABNORMAL_DIR = "bad"  # 异常样本目录名

if __name__ == "__main__":
    datamodule = Folder(
        name="my_dataset",
        root=DATASET_ROOT,
        normal_dir=NORMAL_DIR,
        abnormal_dir=ABNORMAL_DIR,
        normal_test_dir="test/good",  # 测试正常样本位置
        normal_split_ratio=0.0,  # 全部正常图用于训练,不从 good 切测试集
        val_split_mode=ValSplitMode.SAME_AS_TEST,  # 阈值在测试集上拟合(指标偏乐观);NONE 在当前版本会崩
        # mask_dir="ground_truth",  # 有像素级标注时取消注释(用于 pixel_AUROC 等)
    )

    # pill 上参数扫描验证过的最优配置:coreset_sampling_ratio=0.01
    model = Patchcore(coreset_sampling_ratio=0.01)

    engine = Engine(max_epochs=1)

    # 1. 训练
    engine.fit(model=model, datamodule=datamodule)

    # 2. 测试:返回 [指标字典](image_AUROC / image_F1Score / pixel_*)
    results = engine.test(model=model, datamodule=datamodule)
    if results:
        print("\n===== Lightning 记录的测试指标 =====")
        for name, value in results[0].items():
            value = value.item() if hasattr(value, "item") else value
            print(f"{name}: {value:.4f}")

    # 3. 逐图预测,聚合混淆矩阵指标
    predictions = engine.predict(model=model, datamodule=datamodule)
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

    print("\n===== 图像级指标 =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
