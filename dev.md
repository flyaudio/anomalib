# 开发记录(2026-11-13)

Anomalib + YOLO11 骨干的 PatchCore 实验总结。环境:Python 3.14(py314 conda env)、RTX 5090 32GB、anomalib 源码 editable 安装、ultralytics 8.4。

---

## 一、环境搭建要点

1. **安装**:`pip install -e .`(py314 环境原本没有 anomalib);数据在 `~/.cache/anomalib/datasets/MVTecAD/`(pill、screw)
2. **MVTecAD 不会自动下载**(上游链接失效,issue #3892),需手动从 mvtec.com 下载解压
3. **HuggingFace 权重下载**:国内直连会 Connection reset,统一 `export HF_ENDPOINT=https://hf-mirror.com`(建议写入 `~/.bashrc`)
4. **Python 3.14 的坑**:multiprocessing 默认 start method 是 `forkserver`,任何用 DataLoader 多进程的脚本**必须有 `if __name__ == "__main__":` 守卫**(子进程会重新 import 主脚本)。anomalib 官方 examples 都没有守卫,在 3.14 下必崩,已本地修复
5. 其他坑:`anomalib/utils/types/` 包名遮蔽标准库 `types`(不要在 `src/anomalib/utils/` 下直接跑脚本);MVTecAD 源文件权限 550,复制时同名文件覆盖会 PermissionError

## 二、生成的脚本

| 脚本 | 作用 |
|---|---|
| `patchcore_pill_eval.py` | pill 上 PatchCore 训练+测试+混淆矩阵指标+热力图保存 demo |
| `patchcore_pill_sweep.py` | 参数扫描(支持命令行指定配置下标) |
| `patchcore_pill_verify.py` | 多 seed 稳健性验证 |
| `custom_folder_eval.py` | **自建数据集评估模板**(Folder 结构已对照源码修正) |
| `yolo_backbone_patchcore.py` | YOLO11 骨干接 PatchCore 的多模型对比(n/m/l 可扩展) |
| `yolo_finetune_patchcore.py` | 路线 1:OK/NG 二分类微调 YOLO 骨干 → 接回 PatchCore(无泄漏设计) |
| `cross_category_test.py` | 跨类别区分实验(pill vs screw 双向) |
| `embedding_viewer.py` | embedding 交互式 3D 可视化(Rerun):旋转/平移、点↔图双向联动、框选看图 |

## 三、实验结果

### 1. pill 基准 + 参数扫描(10 组实验)

- 基线(wide_resnet50_2, 默认参数):image_AUROC **0.94**,ACC 0.92
- **最优单项调整:`coreset_sampling_ratio=0.01`**(默认 0.1,原论文就是 1%)
  - 3 seed 验证稳健:AUROC **0.960~0.967**,ACC 0.934~0.946,漏报 7 → 3~5 张
- `num_neighbors=25`、`layers L1+L2+L3` 有次级增益但组合无叠加;`backbone=resnet50` 明显变差
- 教训:accuracy 对这些调整不敏感(全卡在 0.9162),**对比实验看 AUROC**

### 2. 热力图可视化

- anomalib 内置 `anomalib.visualization.ImageVisualizer`,支持 anomaly_map/pred_mask 叠加,模型自带 `model.visualizer`
- `engine.test` 返回的是**指标字典**(1 dict/每 dataloader),逐图预测要用 `engine.predict`(默认走 test_dataloader,输出含 pred_score/pred_label/gt_label)
- demo 已集成进 `patchcore_pill_eval.py`,输出 `results/heatmaps/`(512×256 原图+热力图并排)

### 3. YOLO11 骨干直接替换(类内缺陷检测)

| 骨干 | P3/P4 通道 | image_AUROC | 混淆矩阵(TN/FP/FN/TP) |
|---|---|---|---|
| wide_resnet50_2(ImageNet) | 512/1024 | **0.9624** | 22/4/7/134 |
| yolo11n(COCO) | 128/128 | 0.8036 | 1/25/1/140 |
| yolo11m(COCO) | 512/512 | 0.8625 | 8/18/4/137 |

- COCO 特征模式:**高召回(几乎不漏)、低特异(正常图大量误报)**——检测预训练对纹理不敏感,与 PatchCore"正常紧凑/缺陷发散"假设相反
- 通道数有用(n→m:+0.06 AUROC),但追不上 ImageNet 骨干;**决定效果的是预训练任务,不是架构**
- 技术要点:YOLO 骨干走 `TimmFeatureExtractor` 的 nn.Module + TorchFX 路径,需给 `C2f.forward` 打 FX 兼容补丁(数值等价);`pre_processor=False` + 骨干入口 ×255

### 4. 跨类别区分(pill vs screw)

| 方向 | yolo11m AUROC | wr50_2 AUROC |
|---|---|---|
| train=pill → 区分 screw | **1.0000** | 1.0000 |
| train=screw → 区分 pill | **1.0000** | 1.0000 |

- **COCO 特征跨类别满分,与 ImageNet 打平**:检测预训练擅长"物体身份"编码
- 产线含义:异物/错误零件混入类缺陷,yolo 骨干 PatchCore 即可胜任;表面纹理缺陷不行

### 5. 路线 1:OK/NG 二分类微调 YOLO 骨干(无泄漏设计)

设计:微调用 OK + 4 种缺陷类型;**评测用 3 种未见过缺陷类型**(faulty_imprint/pill_type/scratch),测对新缺陷的泛化。

| 骨干 | AUROC | FP |
|---|---|---|
| wr50_2 上界 | 0.9867 | 3 |
| yolo11n COCO | 0.6760 | 24 |
| yolo11n 微调(最终修复版) | 0.7648 | 19 |

**关键教训(按发现顺序):**
1. **ultralytics 加载的权重全部 `requires_grad=False`**——前两轮"微调"权重一字未动(63 个骨干参数全 False,只有分类头在学),分类器退化成常数预测(训练准确率恰好 = OK 占比 0.7474 是铁证)。修复:训练前显式 `requires_grad=True` + 补上 256 resize(之前一直在 800×800 上训练)
2. 有趣的对照:权重没动但 BN 统计量被动更新的那轮,PatchCore AUROC 反而 **0.8217** > 真实微调的 0.7648——小数据下判别式微调容易过拟合帮倒忙(训练 acc 0.88 / 验证 acc 0.35),训练纪律比"微调与否"更重要
3. 结论:路线 1 机制成立(+0.09 AUROC)但简单配方效果有限;**更推荐 RealNet 式重建预训练(只用 OK,连 NG 都不需要,无泄漏)**或精心调参的判别式微调;用真实产线数据小规模验证最有说服力

## 四、重要结论(可直接用于产线)

1. **推荐组合**:PatchCore(wide_resnet50_2, coreset 0.01)无监督粗筛 + YOLO11 有监督精检,各用各的最优预训练
2. **Folder 数据集的正确结构**(已对照源码验证,`test/bad` 会被忽略!):

```
datasets/my_dataset/
├── good/              # 正常训练图
├── bad/               # 异常图(测试)——异常测试图来自这里
├── test/good/         # 正常测试图(normal_test_dir="test/good")
└── ground_truth/bad/  # 可选掩码(pixel_AUROC 用)
```

   配 `normal_split_ratio=0.0` + `val_split_mode=SAME_AS_TEST`(`NONE` 当前版本会崩)
3. 无泄漏评测纪律:微调用的 NG 类型不能进测试集;正确评测 = 用已知缺陷类型训练,测**未见过类型**的检出
4. `engine.test` 返回 `[指标字典]`;逐图结果用 `engine.predict`;阈值相关指标(ACC/P/R/F1)在 SAME_AS_TEST 下偏乐观,跨实验对比以 AUROC 为准
