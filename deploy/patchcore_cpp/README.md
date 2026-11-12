# PatchCore × TensorRT 10 × C++ 部署参考实现

目标平台：**NVIDIA RTX 5090（Blackwell, sm_120）+ TensorRT 10.11.0.33**。

anomalib 官方不提供 C++ / TensorRT 部署代码，本目录是一套最小可跑的参考实现：
把 PatchCore（含 memory bank 最近邻检索、异常图、后处理）整体导出为单个 ONNX，
在 TensorRT 里整图加速，C++ 端只负责图像加载与可视化。

## 文件清单

| 文件 | 作用 |
|---|---|
| `export_patchcore_onnx.py` | 训练（或加载 ckpt）PatchCore 并导出 ONNX（FP32、动态 batch、静态 224×224） |
| `build_engine.ps1` / `build_engine.sh` | 用 `trtexec` 把 ONNX 构建为 FP16 TensorRT 引擎 |
| `main.cpp` | TensorRT 10 C++ 推理：预处理（OpenCV）→ 引擎推理 → 热力图/掩码可视化 |
| `CMakeLists.txt` | C++ 工程构建文件 |

## 关键设计（为什么这么切分）

- **预处理（resize/crop/normalize）放在 C++**：anomalib 导出的 `Resize` 带
  `antialias=True`，TensorRT 的 ONNX 解析器不支持，放图外更稳。
- **模型主体 + 后处理放在图内**：PatchCore 的 `memory_bank` 是 `register_buffer`，
  导出后作为常量权重打进 ONNX；最近邻检索是手写 matmul 实现的欧氏距离
  （源码注释明确说明这是为了 ONNX/OpenVINO 兼容），天然适配 TensorRT。
  后处理（分数归一化、自适应阈值）也是纯张量运算，一并入图，
  因此**输出已归一化到 [0,1]，0.5 即验证集上学习到的自适应阈值**。
- **FP32 导出、FP16 构建引擎**：FP16 由 TensorRT 在构建时处理，避免训练侧精度问题。
  ⚠️ **不建议 INT8**：最近邻距离对量化敏感，容易明显掉点。

## 环境要求

- GPU：RTX 5090（sm_120）；**引擎必须在本机构建，不可跨 GPU 复制**
- TensorRT 10.11.0.33（解压后设 `TRT_DIR` 环境变量）
- CUDA Toolkit 与 TensorRT 10.11 包匹配（12.9 或 13.0，看 zip 包后缀）
- Python 侧：anomalib + ONNX 导出依赖

```bash
uv sync --extra openvino        # 仓库根目录（openvino extra 自带 onnxscript / onnx）
# 或： pip install "anomalib[openvino]"
```

- C++ 侧：CMake ≥ 3.18、OpenCV 4.x、对应 CUDA Toolkit

## 步骤

### 1. 导出 ONNX（Python，无需 GPU 也行，有 GPU 更快）

```bash
# 方式 A：从头训练 MVTec AD bottle 类并导出（需要数据集）
python deploy/patchcore_cpp/export_patchcore_onnx.py \
    --data_root ./datasets/MVTecAD --category bottle

# 方式 B：已有训练好的 checkpoint，直接导出
python deploy/patchcore_cpp/export_patchcore_onnx.py \
    --ckpt_path results/patchcore/mvtecad/bottle/weights/lightning/model.ckpt
```

产物：
- `exports/weights/onnx/patchcore.onnx` —— 图签名：输入 `input` (N,3,224,224) FP32；
  输出 `pred_score`、`anomaly_map`、`pred_label`、`pred_mask`
- `exports/weights/onnx/patchcore.stats.json` —— 后处理阈值/归一化统计（调试参考）

注意 memory bank 会内嵌进 ONNX：以 MVTec bottle 为例（209 张训练图 × 784 patch ×
1536 维，coreset 10%），约 **96 MB** 常量权重。可用 `--coreset_sampling_ratio 0.05`
进一步压缩（也直接加快最近邻检索）。

### 2. 构建 TensorRT 引擎（在 RTX 5090 机器上）

```powershell
# Windows
$env:TRT_DIR = "D:\TensorRT-10.11.0.33"
.\deploy\patchcore_cpp\build_engine.ps1
```

```bash
# Linux
TRT_DIR=/opt/TensorRT-10.11.0.33 bash deploy/patchcore_cpp/build_engine.sh
```

得到 `patchcore_5090_fp16.engine`。构建脚本里 batch 上限设为 8，按需改 `--maxShapes`。

### 3. 编译 C++

```powershell
cmake -S deploy/patchcore_cpp -B build ^
      -DTRT_DIR="D:/TensorRT-10.11.0.33" -DOpenCV_DIR="D:/opencv/build"
cmake --build build --config Release
```

```bash
cmake -S deploy/patchcore_cpp -B build -DTRT_DIR=/opt/TensorRT-10.11.0.33
cmake --build build -j
```

### 4. 推理

```bash
patchcore_trt patchcore_5090_fp16.engine datasets/MVTecAD/bottle/test/broken_large --out ./results
```

输出：每张图的 `pred_score`（已归一化）、判定结果、耗时，以及保存到 `--out` 的
热力图叠加 + 红色异常轮廓可视化图。`pred_label` 直接给出是否异常；
若用 `--no_post_processor` 导出的裸分数，则用 `--threshold` 手动指定阈值。

## 常见问题

- **deserialize 失败**：引擎与 GPU/驱动/TRT 版本不匹配 → 在目标机上重新构建。
- **trtexec 报 Resize/antialias**：说明导出时没去掉 `pre_processor`，检查用的是
  本目录的导出脚本。
- **分数分布与 Python 端略有差异**：OpenCV `INTER_AREA` 与 torchvision antialias
  双线性有像素级近似差，对分数影响很小；如需严格一致，可在 C++ 复现
  torchvision 的抗锯齿滤波器。
- **想要更低延迟**：`--coreset_sampling_ratio` 调小（如 0.01~0.05），或换
  `--backbone resnet18`（论文指标略降）。

## 进阶路线：特征提取与最近邻拆分

若 memory bank 很大、想按需换库或省显存，可采用两段式（本目录未含实现）：
backbone+AvgPool 部分单独导 ONNX 转 TensorRT；`memory_bank` 用
`model.model.memory_bank.cpu().numpy()` 存成 `.npy`，C++ 端加载后用
FAISS（C++/GPU）做最近邻，上采样+高斯平滑+打分段用 OpenCV 复现
（对应 `anomaly_map.py`，就一个 interpolate + σ=4 高斯模糊）。
