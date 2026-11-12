# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""通过"可执行的测试"逐行讲解 ``PatchcoreModel`` 的每个成员函数。

本文件与 ``torch_model.py`` 放在同一目录，既是一份**单元测试**，也是一份**教程**。
每一个成员函数（``__init__`` / ``forward`` / ``generate_embedding`` /
``reshape_embedding`` / ``subsample_embedding`` / ``euclidean_dist`` /
``nearest_neighbors`` / ``compute_anomaly_score``）都有一组对应的测试函数，
每个测试函数的 docstring 就是对该函数"某一个具体功能点"的讲解。

运行方式
--------
两种方式都可以（推荐第一种，不需要额外装 pytest）::

    # 1) 直接当脚本跑（自带极简 test runner）
    python src/anomalib/models/image/patchcore/torch_model_test.py

    # 只跑名字里含某关键字的测试，例如只看 euclidean_dist 相关::
    python src/anomalib/models/image/patchcore/torch_model_test.py euclidean

    # 2) 用 pytest 跑
    pytest src/anomalib/models/image/patchcore/torch_model_test.py -v

为什么用 resnet18 + 128x128 + pre_trained=False
----------------------------------------------
PatchCore 默认用 ``wide_resnet50_2``（需要下载 ~250MB 预训练权重、推理很慢）。
单元测试只关心**张量形状与数值语义**，所以用最小的 resnet18、128x128 输入、
随机初始化权重（``pre_trained=False``），保证离线可跑且秒级完成。
这不会改变任何被测逻辑：所有函数都与骨干种类无关。

被测试的成员函数总览
--------------------
=========================  ==========================================================
成员函数                    职责一句话概括
=========================  ==========================================================
``__init__``               搭积木：骨干(冻结) + 平均池化 + 空的 memory bank + 空仓库
``forward``                总流程：提特征 -> 拼 embedding -> 训练则入库/推理则打分
``generate_embedding``     多尺度特征对齐分辨率后沿通道维拼接
``reshape_embedding``      4D 特征图 (B,C,H,W) 拉平成 2D patch 矩阵 (B*H*W, C)
``subsample_embedding``    coreset 采样：把仓库里的特征筛出一小部分写进 memory bank
``euclidean_dist``         手写欧氏距离矩阵（不用 cdist，为 ONNX 导出让路）
``nearest_neighbors``      分块暴力检索最近的 k 个 memory bank 向量
``compute_anomaly_score``  论文里的 softmax 加权重打分（图像级分数）
=========================  ==========================================================
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.utils import _pair

from anomalib.data import InferenceBatch
from anomalib.data.utils.tiler import Tiler
from anomalib.models.image.patchcore import torch_model as torch_model_module
from anomalib.models.image.patchcore.torch_model import DEFAULT_CHUNK_SIZE, PatchcoreModel

# --------------------------------------------------------------------------------------
# 测试用的轻量配置（不要改成 wide_resnet50_2，否则测试会又慢又要下载权重）
# --------------------------------------------------------------------------------------
TEST_BACKBONE = "resnet18"
TEST_LAYERS = ["layer1", "layer2"]  # 通道数 64 + 128 = 192
TEST_IMG_SIZE = 128  # layer1 -> 32x32, layer2 -> 16x16
TEST_EMBED_DIM = 192  # 64 + 128
TEST_PATCH_GRID = 32  # embedding 的空间分辨率 = 32x32
TEST_N_PATCHES_PER_IMAGE = TEST_PATCH_GRID * TEST_PATCH_GRID  # 1024

torch.manual_seed(42)


# ======================================================================================
# 公共辅助函数
# ======================================================================================
def make_model(
    layers: list[str] | None = None,
    num_neighbors: int = 9,
) -> PatchcoreModel:
    """构造一个轻量、可复现的 PatchcoreModel。

    用 resnet18 + 随机权重，避免下载预训练权重；返回后默认处于 ``train()`` 模式之外，
    由各测试自行决定 ``train()`` / ``eval()``。
    """
    return PatchcoreModel(
        backbone=TEST_BACKBONE,
        layers=layers if layers is not None else list(TEST_LAYERS),
        pre_trained=False,
        num_neighbors=num_neighbors,
    )


def random_images(n: int, size: int = TEST_IMG_SIZE) -> torch.Tensor:
    """生成 n 张随机 (3,size,size) 图片。"""
    generator = torch.Generator().manual_seed(1234 + n)
    return torch.rand(n, 3, size, size, generator=generator)


def fill_memory_bank(model: PatchcoreModel, n_rows: int, dim: int = TEST_EMBED_DIM) -> torch.Tensor:
    """给模型塞一个人造的 memory bank，返回这个 bank 本身。

    PatchCore 的打分逻辑完全不关心 memory bank 是怎么来的（coreset 采样只是"更好"，
    不是"必须"），所以单元测试里直接塞随机向量，把注意力集中在被测函数上。
    """
    generator = torch.Generator().manual_seed(7)
    bank = torch.rand(n_rows, dim, generator=generator)
    model.memory_bank = bank
    return bank


# ======================================================================================
# 1. __init__ —— 构造函数：模型由哪些零件组成
# ======================================================================================
def test_init_stores_config_and_submodules() -> None:
    """功能点 1：保存超参，并搭好 4 个零件。

    ``__init__`` 里依次创建了：
      * ``tiler``              —— 大图切块，默认 ``None``（不切）
      * ``feature_extractor``  —— timm 骨干，且立刻 ``.eval()`` 冻结
      * ``feature_pooler``     —— ``AvgPool2d(3,1,1)`` 邻域平滑（不改变尺寸）
      * ``anomaly_map_generator`` —— patch 分数 -> 热力图
    """
    model = make_model()
    assert model.tiler is None, "默认不启用切块"
    assert model.backbone == TEST_BACKBONE
    assert model.layers == TEST_LAYERS
    assert model.num_neighbors == 9, "默认取 9 个最近邻"
    assert isinstance(model.feature_pooler, nn.AvgPool2d)
    assert model.anomaly_map_generator is not None


def test_init_feature_extractor_is_frozen_and_in_eval_mode() -> None:
    """功能点 2：骨干是"冻结的照相机"，不参与训练。

    源码 ``TimmFeatureExtractor(...).eval()`` 做了两件事：
      * ``.eval()``                 —— 关掉 dropout、BN 改用滑动统计量
      * ``requires_grad=False``（默认值，保存在 ``self.requires_grad`` 上）

    ⚠️ 一个容易误解的细节：``requires_grad=False`` **并不会**把参数的
    ``Tensor.requires_grad`` 置为 False（源码里没有 ``p.requires_grad_()``）。
    它真正的作用体现在 ``TimmFeatureExtractor.forward`` 里——
    ``self.feature_extractor.eval()`` + ``with torch.no_grad()``，
    也就是**前向时压根不建计算图**。配合 ``torch_model.forward`` 外层那个
    ``with torch.no_grad()``，梯度无从产生，"无训练"由此成立。
    """
    model = make_model()
    assert not model.feature_extractor.training, "构造后骨干必须处于 eval 模式"
    assert model.feature_extractor.requires_grad is False, "默认不计算骨干梯度"

    # 验证 no_grad 是真的生效：把模型切到 train 模式，输出依然不带梯度
    model.train()
    assert model.feature_extractor.training is True, "nn.Module.train() 会递归影响子模块"
    out = model(random_images(1))
    assert not out.requires_grad, "即使 train 模式，骨干前向仍在 no_grad 下执行"


def test_init_feature_pooler_preserves_spatial_size() -> None:
    """功能点 3：``AvgPool2d(3, 1, 1)`` 只做 3x3 邻域平滑，分辨率不变。

    它对每个 patch 取 3x3 邻域平均，抑制局部噪声；padding=1 保证 H/W 不变。
    """
    model = make_model()
    pooler = model.feature_pooler
    assert _pair(pooler.kernel_size) == (3, 3)
    assert _pair(pooler.stride) == (1, 1)
    assert _pair(pooler.padding) == (1, 1)
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    assert pooler(x).shape == x.shape, "池化后尺寸必须与输入一致"


def test_init_memory_bank_and_store_are_empty() -> None:
    """功能点 4：初始状态 memory bank 为空、embedding 仓库为空列表。

    ``register_buffer("memory_bank", torch.empty(0))`` —— 空的 (0,) 张量；
    ``embedding_store`` 是一个普通 Python list，训练时不断 append。
    """
    model = make_model()
    assert model.memory_bank.numel() == 0, "初始 memory bank 必须是空的"
    assert model.embedding_store == [], "初始 embedding 仓库必须是空列表"


def test_init_memory_bank_is_a_buffer_not_a_parameter() -> None:
    """功能点 5：memory bank 是 **buffer** 而不是 parameter —— 这是导出部署的关键。

    * 不是 parameter  => 不会被优化器更新（再次印证"无训练"）
    * 是 buffer       => 会出现在 ``state_dict`` 里，能随 checkpoint 存盘
    * 是 buffer       => ``torch.onnx.export`` 时会作为**图内常量**打进 ONNX
    最后一条正是我们之前做 TensorRT 部署时 memory bank 能被内嵌进计算图的原因。
    """
    model = make_model()
    buffer_names = dict(model.named_buffers())
    param_names = dict(model.named_parameters())
    assert "memory_bank" in buffer_names
    assert "memory_bank" not in param_names


def test_init_dynamic_buffer_allows_loading_different_bank_size() -> None:
    """功能点 6：继承 ``DynamicBufferMixin`` —— 允许加载尺寸不同的 memory bank。

    普通 ``nn.Module`` 加载形状不匹配的 buffer 会直接报错；PatchcoreModel 会在
    ``_load_from_state_dict`` 里先把 buffer ``resize_`` 成新形状再写入。
    所以"用 A 数据集训的 bank 大小"和"当前模型"不一致时也能 load_state_dict。
    """
    model = make_model()
    state_dict = model.state_dict()
    state_dict["memory_bank"] = torch.rand(37, TEST_EMBED_DIM)
    model.load_state_dict(state_dict)
    assert model.memory_bank.shape == (37, TEST_EMBED_DIM), "buffer 应被自动 resize 成 (37, 192)"


# ======================================================================================
# 2. generate_embedding —— 多尺度特征拼接
# ======================================================================================
def test_generate_embedding_single_layer_is_passthrough() -> None:
    """功能点 1：只有一层时，直接返回该层特征（无需拼接、无需上采样）。"""
    model = make_model(layers=["layer1"])
    feat = torch.rand(2, 64, 8, 8)
    assert torch.equal(model.generate_embedding({"layer1": feat}), feat)


def test_generate_embedding_concatenates_channels() -> None:
    """功能点 2：多层特征沿 **通道维(dim=1)** 拼接，通道数相加。

    layer1: (B,64,32,32) + layer2: (B,128,16,16) -> (B,192,32,32)
    """
    model = make_model()
    features = {
        "layer1": torch.rand(2, 64, 32, 32),
        "layer2": torch.rand(2, 128, 16, 16),
    }
    embedding = model.generate_embedding(features)
    assert embedding.shape == (2, 192, 32, 32)


def test_generate_embedding_upsamples_to_first_layer_size() -> None:
    """功能点 3：所有层都被上采样到 **layers[0]** 的空间尺寸。

    源码 ``F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")``
    也就是以第一层（分辨率最高）为基准网格。这也是为什么最终 patch 网格是 32x32
    而不是 layer2 的 16x16——用高分辨率网格才能定位到更细的缺陷。
    """
    model = make_model()
    features = {
        "layer1": torch.rand(1, 64, 32, 32),
        "layer2": torch.rand(1, 128, 16, 16),
        "layer3": torch.rand(1, 256, 8, 8),
    }
    model.layers = ["layer1", "layer2", "layer3"]
    embedding = model.generate_embedding(features)
    assert embedding.shape[-2:] == (32, 32), "空间尺寸必须对齐到第一层"
    assert embedding.shape[1] == 64 + 128 + 256


def test_generate_embedding_matches_manual_bilinear() -> None:
    """功能点 4：拼接结果 == 手工 ``F.interpolate`` + ``torch.cat`` 的结果。"""
    model = make_model()
    layer1 = torch.rand(1, 64, 32, 32)
    layer2 = torch.rand(1, 128, 16, 16)
    expected = torch.cat(
        (layer1, F.interpolate(layer2, size=layer1.shape[-2:], mode="bilinear")),
        dim=1,
    )
    torch.testing.assert_close(model.generate_embedding({"layer1": layer1, "layer2": layer2}), expected)


# ======================================================================================
# 3. reshape_embedding —— 4D 特征图 -> 2D patch 矩阵
# ======================================================================================
def test_reshape_embedding_flattens_spatial_dims() -> None:
    """功能点 1：形状从 (B,C,H,W) 变成 (B*H*W, C)，每一行是一个 patch 的向量。"""
    embedding = torch.rand(4, 192, 32, 32)
    reshaped = PatchcoreModel.reshape_embedding(embedding)
    assert reshaped.shape == (4 * 32 * 32, 192)


def test_reshape_embedding_row_order_is_nhwc_flatten() -> None:
    """功能点 2：行顺序是 **按 NHWC 逐行展开**（先 W 后 H），不是随便 flatten。

    源码 ``embedding.permute(0,2,3,1).reshape(-1, C)``：
    permute 成 (B,H,W,C) 后展平，所以同一张图的 patch 是**行优先**排列的。
    这个顺序必须和后面 ``patch_scores.reshape((B,1,W,H))`` 严格互逆，
    否则热力图会"错位"。这里用双重 for 循环手工对照以证明顺序正确。
    """
    generator = torch.Generator().manual_seed(0)
    embedding = torch.rand(2, 5, 3, 4, generator=generator)  # (B=2, C=5, H=3, W=4)
    reshaped = PatchcoreModel.reshape_embedding(embedding)

    assert reshaped.shape == (2 * 3 * 4, 5)
    for b in range(2):
        for h in range(3):
            for w in range(4):
                row = b * (3 * 4) + h * 4 + w
                torch.testing.assert_close(reshaped[row], embedding[b, :, h, w])


def test_reshape_embedding_is_inverse_of_score_reshape() -> None:
    """功能点 3：验证 reshape 与 forward 里"分数还原成图"互为逆操作。

    forward 里先把 (B,C,H,W) 拉平成 (B*H*W, C)，打完分后再
    ``patch_scores.reshape((B,1,W,H))``。两者必须严格对应，热力图才不会转置。
    """
    batch, height, width = 2, 8, 6
    anomaly_scores = torch.arange(batch * height * width, dtype=torch.float32)
    as_image = anomaly_scores.reshape(batch, 1, height, width)
    assert torch.equal(as_image.flatten(), anomaly_scores), "还原后展平必须与原分数一致"
    # 第 b 张图第 (h,w) 个位置的分数，就是拉平后第 b*H*W + h*W + w 个元素
    assert as_image[1, 0, 2, 3] == anomaly_scores[1 * height * width + 2 * width + 3]


# ======================================================================================
# 4. euclidean_dist —— 手写欧氏距离矩阵（ONNX 友好）
# ======================================================================================
def test_euclidean_dist_matches_torch_cdist() -> None:
    """功能点 1：结果与 ``torch.cdist`` 完全一致，但实现只用矩阵乘。

    源码用了恒等式  ||x-y||^2 = ||x||^2 - 2*x·y + ||y||^2，
    全程 ``matmul`` / ``add`` / ``clamp_min`` / ``sqrt``，
    注释里写明是为了 "better compatibility with ONNX export and OpenVINO conversion"。
    """
    generator = torch.Generator().manual_seed(1)
    x = torch.rand(37, 64, generator=generator)
    y = torch.rand(53, 64, generator=generator)
    torch.testing.assert_close(PatchcoreModel.euclidean_dist(x, y), torch.cdist(x, y))


def test_euclidean_dist_hand_computed_values() -> None:
    """功能点 2：手算验证。3 维空间里两个点的距离应该完全对得上。"""
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y = torch.tensor([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
    dist = PatchcoreModel.euclidean_dist(x, y)
    # x0->y0: 5, x0->y1: 1 ; x1->y0: sqrt(4+16)=4.4721, x1->y1: 0
    expected = torch.tensor([[5.0, 1.0], [4.4721, 0.0]])
    torch.testing.assert_close(dist, expected, rtol=1e-4, atol=1e-4)


def test_euclidean_dist_self_distance_is_zero() -> None:
    """功能点 3：同一个向量和自己的距离为 0（对角线为 0）。

    用整数向量而非随机浮点：浮点下 ``||x||^2 - 2x·x + ||x||^2`` 会残留 ~1e-7 的误差，
    开根号后可能放大成 ~1e-4（这也正是源码要 ``clamp_min_(0)`` 的原因）。
    整数能保证中间结果精确相消，对角线严格为 0。
    """
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.0, 2.0]])
    dist = PatchcoreModel.euclidean_dist(x, x)
    assert dist.shape == (3, 3)
    assert torch.equal(dist.diag(), torch.zeros(3)), "自己到自己的距离必须为 0"
    assert torch.equal(dist, dist.T), "距离矩阵必须对称"


def test_euclidean_dist_is_symmetric() -> None:
    """功能点 4：距离矩阵对称 —— d(i,j) == d(j,i)。"""
    x = torch.rand(11, 32)
    y = torch.rand(11, 32)
    torch.testing.assert_close(PatchcoreModel.euclidean_dist(x, y), PatchcoreModel.euclidean_dist(y, x).T)


def test_euclidean_dist_clamps_negative_to_zero() -> None:
    """功能点 5：``clamp_min_(0)`` 兜底浮点误差，保证根号内非负、结果不出 NaN。

    展开式 ``||x||^2 - 2xy + ||y||^2`` 在浮点下可能算出极小负数：
    当查询向量与 memory bank 里某个向量**完全相同**时，理论值 0 实际可能是 -1e-8，
    直接 ``sqrt_`` 会得到 NaN。源码用 ``clamp_min_(0)`` 挡住这种情况。
    """
    # 1) 相同向量：最容易触发 -0.0 量级的负残差
    same = torch.tensor([[0.1, 0.2, 0.3]])
    dist = PatchcoreModel.euclidean_dist(same, same)
    assert torch.isfinite(dist).all(), "相同向量的距离不能是 NaN"
    # 若没有 clamp_min_(0)，这里会是 NaN（NaN 不满足 >= 0，断言会失败）
    assert 0.0 <= dist.item() < 1e-3, f"相同向量距离应≈0，实际 {dist.item()}"

    # 2) 极端量级：中间结果 ~1e16，仍不能溢出成 inf / NaN
    x = torch.tensor([[1e8, 1.0]])
    y = torch.tensor([[1.0, 1e8]])
    dist = PatchcoreModel.euclidean_dist(x, y)
    assert torch.isfinite(dist).all(), "不能出现 inf/NaN"
    assert (dist >= 0).all(), "距离必须非负"


# ======================================================================================
# 5. nearest_neighbors —— 分块暴力 kNN 检索
# ======================================================================================
def test_nearest_neighbors_finds_true_nearest() -> None:
    """功能点 1：返回的最近邻确实是 memory bank 里最近的那个（与 cdist 对照）。"""
    model = make_model()
    bank = fill_memory_bank(model, n_rows=200)
    generator = torch.Generator().manual_seed(3)
    queries = torch.rand(30, TEST_EMBED_DIM, generator=generator)

    scores, locations = model.nearest_neighbors(queries, n_neighbors=1)

    reference = torch.cdist(queries, bank)
    expected_scores = reference.min(dim=1).values
    expected_locations = reference.argmin(dim=1)
    # cdist 与手写展开式的累加顺序不同，允许很小的数值差
    torch.testing.assert_close(scores.squeeze(1), expected_scores, atol=1e-4, rtol=1e-4)
    assert torch.equal(locations.squeeze(1), expected_locations)


def test_nearest_neighbors_k1_uses_min_fast_path() -> None:
    """功能点 2：``n_neighbors == 1`` 时走 ``distances.min(1)`` 快路径。

    源码注释："when n_neighbors is 1, speed up computation by using min instead of topk"。
    """
    model = make_model()
    bank = fill_memory_bank(model, n_rows=150)
    queries = torch.rand(10, TEST_EMBED_DIM)
    scores, locations = model.nearest_neighbors(queries, n_neighbors=1)
    assert scores.shape == (10, 1) and locations.shape == (10, 1)
    distances = PatchcoreModel.euclidean_dist(queries, bank)
    torch.testing.assert_close(scores.squeeze(1), distances.min(1).values)
    assert torch.equal(locations.squeeze(1), distances.min(1).indices)


def test_nearest_neighbors_topk_returns_sorted_k_results() -> None:
    """功能点 3：``n_neighbors > 1`` 时走 ``topk(largest=False)``，结果升序且含 k 个。

    并且第 0 列（最近的）必须和 k=1 的结果完全一致。
    """
    model = make_model()
    bank = fill_memory_bank(model, n_rows=300)
    queries = torch.rand(20, TEST_EMBED_DIM)

    scores_k5, locations_k5 = model.nearest_neighbors(queries, n_neighbors=5)
    scores_k1, locations_k1 = model.nearest_neighbors(queries, n_neighbors=1)

    assert scores_k5.shape == (20, 5) and locations_k5.shape == (20, 5)
    # 升序：每一行的距离应单调不减
    assert torch.all(scores_k5[:, 1:] >= scores_k5[:, :-1]), "topk 结果应按距离升序"
    torch.testing.assert_close(scores_k5[:, 0], scores_k1.squeeze(1), atol=1e-5, rtol=1e-5)
    assert torch.equal(locations_k5[:, 0], locations_k1.squeeze(1))


def test_nearest_neighbors_chunked_path_matches_unchunked() -> None:
    """功能点 4：查询数 > ``DEFAULT_CHUNK_SIZE`` 时走分块路径，结果必须与不分块一致。

    源码把大查询切成 1024 一块，逐块算完立刻 topk 并 ``del distances``，
    避免 N×M 的巨型距离矩阵一次性驻留显存。
    这里把 ``DEFAULT_CHUNK_SIZE`` 临时改成 8，就能用很小的张量覆盖分块分支。
    """
    model = make_model()
    bank = fill_memory_bank(model, n_rows=100)
    generator = torch.Generator().manual_seed(5)
    queries = torch.rand(25, TEST_EMBED_DIM, generator=generator)

    original_chunk_size = torch_model_module.DEFAULT_CHUNK_SIZE
    try:
        assert original_chunk_size == DEFAULT_CHUNK_SIZE == 1024
        torch_model_module.DEFAULT_CHUNK_SIZE = 8  # 25 个查询 -> 4 块

        scores_chunked, locations_chunked = model.nearest_neighbors(queries, n_neighbors=3)

        torch_model_module.DEFAULT_CHUNK_SIZE = 10_000  # 强制走"不分块"分支
        scores_single, locations_single = model.nearest_neighbors(queries, n_neighbors=3)
    finally:
        torch_model_module.DEFAULT_CHUNK_SIZE = original_chunk_size

    assert scores_chunked.shape == (25, 3)
    torch.testing.assert_close(scores_chunked, scores_single)
    assert torch.equal(locations_chunked, locations_single)


def test_nearest_neighbors_chunk_boundary_is_exact() -> None:
    """功能点 5：分块边界（n 刚好是 chunk_size 的整数倍 / 余数块）不能漏算或重算。"""
    model = make_model()
    bank = fill_memory_bank(model, n_rows=50)
    queries = torch.rand(16, TEST_EMBED_DIM)

    original = torch_model_module.DEFAULT_CHUNK_SIZE
    try:
        torch_model_module.DEFAULT_CHUNK_SIZE = 8  # 16 = 8 + 8，正好整除
        scores, _ = model.nearest_neighbors(queries, n_neighbors=1)
        assert scores.shape == (16, 1)

        torch_model_module.DEFAULT_CHUNK_SIZE = 5  # 16 = 5+5+5+1，最后一个是余数块
        scores_rem, locations_rem = model.nearest_neighbors(queries, n_neighbors=1)
    finally:
        torch_model_module.DEFAULT_CHUNK_SIZE = original

    assert scores_rem.shape == (16, 1)
    torch.testing.assert_close(scores_rem, scores, atol=1e-6)
    assert locations_rem.max().item() < 50, "返回的索引必须在 memory bank 范围内"


# ======================================================================================
# 6. subsample_embedding —— coreset 采样（PatchCore 唯一的"学习"）
# ======================================================================================
def test_subsample_embedding_raises_when_store_empty() -> None:
    """功能点 1：仓库为空时不能采样，直接抛 ``ValueError``。"""
    model = make_model()
    try:
        model.subsample_embedding(sampling_ratio=0.1)
    except ValueError as err:
        assert "Embedding store is empty" in str(err)
    else:
        raise AssertionError("空仓库时必须抛 ValueError")


def test_subsample_embedding_vstacks_store_then_clears_it() -> None:
    """功能点 2：``torch.vstack`` 把分批收集的 embedding 堆成一个大矩阵，然后清空仓库。

    ``embedding_store`` 是 list[Tensor]，每个元素是一批图的所有 patch
    （形状 (B*H*W, C)）；vstack 后得到 (N_total, C)。
    随后 ``self.embedding_store.clear()`` 释放内存——这一步很重要，
    否则 MVTec 级别的数据集会占几十 GB。
    """
    model = make_model()
    model.embedding_store = [torch.rand(1024, TEST_EMBED_DIM), torch.rand(2048, TEST_EMBED_DIM)]
    model.subsample_embedding(sampling_ratio=0.1)
    assert model.embedding_store == [], "采样后仓库必须被清空"
    assert model.memory_bank.shape == (307, TEST_EMBED_DIM)


def test_subsample_embedding_coreset_size_follows_ratio() -> None:
    """功能点 3：memory bank 的行数 = ``int(N * sampling_ratio)``。"""
    model = make_model()
    model.embedding_store = [torch.rand(1000, TEST_EMBED_DIM)]
    model.subsample_embedding(sampling_ratio=0.05)
    assert model.memory_bank.shape[0] == 50, "1000 * 0.05 = 50"
    assert model.memory_bank.shape[1] == TEST_EMBED_DIM


def test_subsample_embedding_coreset_rows_come_from_store() -> None:
    """功能点 4：coreset 采样的每一行都必须真的来自原始 embedding（是"挑"不是"造"）。"""
    generator = torch.Generator().manual_seed(11)
    original = torch.rand(300, 8, generator=generator)
    model = make_model()
    model.embedding_store = [original]
    model.subsample_embedding(sampling_ratio=0.1)  # 选出 30 行

    coreset = model.memory_bank
    assert coreset.shape == (30, 8)
    for row in coreset:
        matches = (original == row).all(dim=1)
        assert matches.any(), "coreset 里出现了原始数据中不存在的向量"


def test_subsample_embedding_ignores_deprecated_embeddings_arg() -> None:
    """功能点 5：已废弃的 ``embeddings`` 参数只触发告警，不影响结果（被 ``del`` 掉）。

    源码 ``@deprecate(args={"embeddings": None}, since="2.1.0", ...)`` 之后
    ``if embeddings is not None: del embeddings`` —— 传什么都无所谓，
    真正的数据源永远是 ``self.embedding_store``。

    注：coreset 采样内部有随机起点，所以这里不对比两次采样的具体结果，
    只验证"传了废弃参数后，行为依然正确"（尺寸 + 来源）。
    """
    generator = torch.Generator().manual_seed(13)
    original = torch.rand(200, 8, generator=generator)

    model = make_model()
    model.embedding_store = [original.clone()]
    # 故意传一个形状完全不同的 embeddings，看它是否真的被忽略
    model.subsample_embedding(sampling_ratio=0.2, embeddings=torch.rand(5, 3))

    assert model.memory_bank.shape == (40, 8), "结果仍应由 embedding_store 决定：200 * 0.2 = 40"
    for row in model.memory_bank:
        assert (original == row).all(dim=1).any(), "coreset 行必须来自 embedding_store"


# ======================================================================================
# 7. compute_anomaly_score —— 论文中的 softmax 加权重打分
# ======================================================================================
def test_compute_anomaly_score_num_neighbors_one_returns_max() -> None:
    """功能点 1：``num_neighbors == 1`` 时直接取最大的 patch 分数（论文原始版打分）。

    源码第一行就是 "Don't need to compute weights if num_neighbors is 1"，
    直接 ``patch_scores.amax(1)``——一张图的异常分 = 最异常的那个 patch 的分数。
    """
    model = make_model(num_neighbors=1)
    fill_memory_bank(model, n_rows=100)
    patch_scores = torch.tensor([[1.0, 5.0, 3.0, 2.0], [0.5, 0.1, 0.9, 0.2]])
    locations = torch.zeros(2, 4, dtype=torch.long)
    embedding = torch.rand(2 * 4, TEST_EMBED_DIM)

    scores = model.compute_anomaly_score(patch_scores, locations, embedding)
    torch.testing.assert_close(scores, torch.tensor([5.0, 0.9]))


def test_compute_anomaly_score_weighted_is_bounded_by_raw_score() -> None:
    """功能点 2：``num_neighbors > 1`` 时加权，结果一定 <= 原始 s*（权重在 [0,1]）。

    论文动机：如果最异常 patch 的最近邻在 memory bank 里"很孤立"，
    那它大概率只是噪声；权重 ``1 - softmax(d)[0]`` 会把分数压低。
    所以加权打分只会**降低**分数，不会放大。
    """
    model = make_model(num_neighbors=9)
    fill_memory_bank(model, n_rows=200)
    generator = torch.Generator().manual_seed(17)
    batch, num_patches = 4, 49
    embedding = torch.rand(batch * num_patches, TEST_EMBED_DIM, generator=generator)
    patch_scores, locations = model.nearest_neighbors(embedding, n_neighbors=1)
    patch_scores = patch_scores.reshape(batch, num_patches)
    locations = locations.reshape(batch, num_patches)

    raw_score = patch_scores.amax(1)
    weighted = model.compute_anomaly_score(patch_scores, locations, embedding)

    assert weighted.shape == (batch,)
    assert torch.all(weighted >= 0), "加权分数不能为负"
    assert torch.all(weighted <= raw_score + 1e-5), "加权后不应超过原始 s*"


def test_compute_anomaly_score_matches_paper_formula() -> None:
    """功能点 3：手工逐步复现论文公式，验证源码 6 个步骤一一对应。

    论文（PatchCore 3.3 节）步骤：
      1. 找每张图里 s* 最大的 patch            -> ``m^{test,*}``
      2. 取它的最近邻距离 s* 与其在 bank 中的下标 -> ``m^*``
      3. 找 m^* 在 bank 里的 n 个支撑样本        -> ``N_b(m^*)``
      4. 算 m^{test,*} 到这些支撑样本的距离
      5. softmax 求权重  w = 1 - softmax(d)[0]
      6. s = w * s*
    """
    num_neighbors = 5
    model = make_model(num_neighbors=num_neighbors)
    bank = fill_memory_bank(model, n_rows=120)
    generator = torch.Generator().manual_seed(19)
    batch, num_patches = 3, 25
    embedding = torch.rand(batch * num_patches, TEST_EMBED_DIM, generator=generator)

    patch_scores, locations = model.nearest_neighbors(embedding, n_neighbors=1)
    patch_scores = patch_scores.reshape(batch, num_patches)
    locations = locations.reshape(batch, num_patches)

    # ---- 手工复现 ----
    max_patches = torch.argmax(patch_scores, dim=1)
    max_patch_features = embedding.reshape(batch, num_patches, -1)[torch.arange(batch), max_patches]
    score_star = patch_scores[torch.arange(batch), max_patches]
    nn_index = locations[torch.arange(batch), max_patches]
    nn_sample = bank[nn_index, :]
    _, support_samples = model.nearest_neighbors(nn_sample, n_neighbors=num_neighbors)
    distances = PatchcoreModel.euclidean_dist(max_patch_features.unsqueeze(1), bank[support_samples])
    weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
    expected = weights * score_star

    actual = model.compute_anomaly_score(patch_scores, locations, embedding)
    torch.testing.assert_close(actual, expected)


def test_compute_anomaly_score_handles_memory_bank_smaller_than_k() -> None:
    """功能点 4：memory bank 行数 < ``num_neighbors`` 时用 ``min()`` 兜底，不会崩。

    源码 ``n_neighbors=min(self.num_neighbors, memory_bank_effective_size)``，
    注释写着 "edge case when memory bank is too small"。
    如果直接传 ``k=9`` 去 topk 一个只有 4 行的距离矩阵会抛 RuntimeError。
    """
    model = make_model(num_neighbors=9)
    fill_memory_bank(model, n_rows=4)  # 故意小于 9
    batch, num_patches = 2, 9
    embedding = torch.rand(batch * num_patches, TEST_EMBED_DIM)
    patch_scores, locations = model.nearest_neighbors(embedding, n_neighbors=1)
    patch_scores = patch_scores.reshape(batch, num_patches)
    locations = locations.reshape(batch, num_patches)

    scores = model.compute_anomaly_score(patch_scores, locations, embedding)
    assert scores.shape == (batch,)
    assert torch.isfinite(scores).all()


# ======================================================================================
# 8. forward —— 总流程（训练态 vs 推理态）
# ======================================================================================
def test_forward_train_mode_returns_embedding_and_stores_it() -> None:
    """功能点 1：``model.train()`` 时 forward 只返回 embedding 并塞进仓库。

    这就是 PatchCore "训练"的全部内容——没有任何 loss / backward。
    返回形状是 (B*H*W, C) = (2*32*32, 192)。
    """
    model = make_model()
    model.train()
    out = model(random_images(2))

    assert isinstance(out, torch.Tensor), "训练态必须返回 Tensor 而不是 InferenceBatch"
    assert out.shape == (2 * TEST_N_PATCHES_PER_IMAGE, TEST_EMBED_DIM)
    assert len(model.embedding_store) == 1, "这批 embedding 应被追加到仓库"
    assert torch.equal(model.embedding_store[0], out)
    assert model.memory_bank.numel() == 0, "训练态不碰 memory bank"


def test_forward_runs_without_gradients() -> None:
    """功能点 2：``with torch.no_grad()`` —— 前向不建计算图，输出不需要梯度。

    这也是"无训练"的直接证据：连计算图都不建，自然谈不上反向传播。
    """
    model = make_model()
    model.train()
    out = model(random_images(1))
    assert not out.requires_grad


def test_forward_eval_raises_when_memory_bank_empty() -> None:
    """功能点 3：没有 memory bank 就推理，抛 ``ValueError("Memory bank is empty")``。

    这个检查在源码里是显式的，防止用户忘记 ``subsample_embedding`` 就跑推理。
    """
    model = make_model()
    model.eval()
    try:
        model(random_images(1))
    except ValueError as err:
        assert "Memory bank is empty" in str(err)
    else:
        raise AssertionError("memory bank 为空时必须抛 ValueError")


def test_forward_eval_returns_inference_batch_with_correct_shapes() -> None:
    """功能点 4：``model.eval()`` 时返回 ``InferenceBatch``，形状全部对齐输入。

      * ``pred_score``  : (B,)        每张图一个分数
      * ``anomaly_map`` : (B,1,H,W)   与输入图同分辨率的热力图
    """
    model = make_model()
    model.train()
    model(random_images(2))
    model.subsample_embedding(sampling_ratio=0.1)
    model.eval()

    out = model(random_images(3, size=TEST_IMG_SIZE))

    assert isinstance(out, InferenceBatch)
    assert out.pred_score is not None and out.pred_score.shape == (3,)
    assert out.anomaly_map is not None
    assert out.anomaly_map.shape == (3, 1, TEST_IMG_SIZE, TEST_IMG_SIZE)


def test_forward_eval_does_not_modify_embedding_store() -> None:
    """功能点 5：推理态不会往仓库里塞东西（``if self.training`` 分支才 append）。"""
    model = make_model()
    model.train()
    model(random_images(2))
    model.subsample_embedding(sampling_ratio=0.1)
    store_after_fit = list(model.embedding_store)
    model.eval()
    model(random_images(2))
    assert model.embedding_store == store_after_fit, "推理不应污染 embedding 仓库"


def test_forward_anomaly_map_matches_input_resolution() -> None:
    """功能点 6：``output_size = input_tensor.shape[-2:]`` 在切块**之前**记录，
    保证热力图始终还原到原始输入分辨率，而不是 patch 网格分辨率。
    """
    model = make_model()
    model.train()
    model(random_images(2))
    model.subsample_embedding(sampling_ratio=0.1)
    model.eval()

    for size in (128, 160):
        out = model(random_images(1, size=size))
        assert out.anomaly_map is not None
        assert out.anomaly_map.shape[-2:] == (size, size), f"输入 {size} 时热力图也应是 {size}"


def test_forward_with_tiler_keeps_same_result_when_one_tile() -> None:
    """功能点 7：``self.tiler`` 分支——tile 尺寸 == 图片尺寸时只有 1 块，结果应与不切块完全一致。

    这里让两个模型**共用同一个 memory bank**（直接赋值拷贝），
    从而把"切块逻辑"和"coreset 采样的随机性"解耦，纯比较 forward 的两条分支。
    """
    images = random_images(2)

    model_no_tiler = make_model()
    model_no_tiler.train()
    model_no_tiler(images)
    model_no_tiler.subsample_embedding(sampling_ratio=0.1)
    model_no_tiler.eval()
    expected = model_no_tiler(images)

    model_tiler = make_model()
    model_tiler.memory_bank = model_no_tiler.memory_bank.clone()  # 复用同一 bank，排除采样随机性
    model_tiler.tiler = Tiler(tile_size=TEST_IMG_SIZE, stride=TEST_IMG_SIZE)
    model_tiler.eval()
    actual = model_tiler(images)

    assert actual.pred_score is not None and expected.pred_score is not None
    assert actual.anomaly_map is not None and expected.anomaly_map is not None
    torch.testing.assert_close(actual.pred_score, expected.pred_score, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, atol=1e-5, rtol=1e-5)


def test_forward_with_tiler_multiple_tiles_reconstructs_full_map() -> None:
    """功能点 8：真的切成多块时，``tiler.untile`` 会把 patch embedding 缝回整图。

    128x128 图 + 64x64 块 + stride 64 -> 每图 4 块；切块只影响中间过程，
    最终 anomaly_map 仍应是 (B,1,128,128)。
    """
    model = make_model()
    model.tiler = Tiler(tile_size=64, stride=64)
    model.train()
    model(random_images(1))
    model.subsample_embedding(sampling_ratio=0.1)
    model.eval()

    out = model(random_images(1))
    assert out.anomaly_map is not None
    assert out.anomaly_map.shape == (1, 1, TEST_IMG_SIZE, TEST_IMG_SIZE)
    assert torch.isfinite(out.anomaly_map).all()


# ======================================================================================
# 9. 端到端：把上面所有函数串起来，验证"异常图得分更高"
# ======================================================================================
def _normal_images(n: int, size: int = TEST_IMG_SIZE) -> torch.Tensor:
    """正常样本：统一的灰色背景 + 极小噪声（彼此几乎一样）。"""
    generator = torch.Generator().manual_seed(99)
    base = torch.full((1, 3, size, size), 0.5)
    noise = 0.01 * torch.randn(n, 3, size, size, generator=generator)
    return (base + noise).clamp(0, 1)


def _anomalous_images(n: int, size: int = TEST_IMG_SIZE) -> torch.Tensor:
    """异常样本：正常背景 + 正中间一块明显的白斑（模拟缺陷）。"""
    images = _normal_images(n, size)
    h0, h1 = size // 4, 3 * size // 4
    w0, w1 = size // 4, 3 * size // 4
    images[:, :, h0:h1, w0:w1] = 1.0
    return images


def _fit_model(model: PatchcoreModel, images: torch.Tensor, ratio: float = 0.1) -> PatchcoreModel:
    """完整走一遍 PatchCore 的"训练"：收集 -> coreset 采样。"""
    model.train()
    for start in range(0, images.shape[0], 2):
        model(images[start : start + 2])
    model.subsample_embedding(sampling_ratio=ratio)
    model.eval()
    return model


def test_end_to_end_anomalous_scores_higher_than_normal() -> None:
    """端到端：用正常图建库，异常图的 pred_score 必须明显高于正常图。

    这条测试把整条链路跑通：
    ``forward(train)`` -> ``subsample_embedding`` -> ``forward(eval)``
    -> ``nearest_neighbors`` -> ``compute_anomaly_score`` -> ``anomaly_map``
    """
    model = make_model(num_neighbors=1)
    train_images = _normal_images(4)
    _fit_model(model, train_images)

    normal_out = model(_normal_images(2))
    anomalous_out = model(_anomalous_images(2))

    assert normal_out.pred_score is not None and anomalous_out.pred_score is not None
    assert torch.all(anomalous_out.pred_score > normal_out.pred_score), (
        f"异常分数 {anomalous_out.pred_score.tolist()} 应高于正常分数 {normal_out.pred_score.tolist()}"
    )


def test_end_to_end_anomaly_map_localizes_the_defect() -> None:
    """端到端：热力图的高亮区域应该落在缺陷所在的位置（图中央）。

    这验证了 ``reshape_embedding`` 与 ``patch_scores.reshape((B,1,W,H))``
    的互逆关系没有把图"转置"——否则高亮会出现在错误的象限。
    """
    model = make_model(num_neighbors=1)
    _fit_model(model, _normal_images(4))
    out = model(_anomalous_images(1))

    assert out.anomaly_map is not None
    heat = out.anomaly_map[0, 0]
    size = heat.shape[0]
    quarter = size // 4
    center_mean = heat[quarter : 3 * quarter, quarter : 3 * quarter].mean()
    border_mean = torch.cat([heat[:quarter].flatten(), heat[-quarter:].flatten()]).mean()
    assert center_mean > border_mean, f"缺陷区域({center_mean:.4f})应比边缘({border_mean:.4f})更热"


def test_end_to_end_works_with_weighted_scoring() -> None:
    """端到端：``num_neighbors=9`` 的加权重打分路径同样能区分正常/异常。"""
    model = make_model(num_neighbors=9)
    _fit_model(model, _normal_images(4))

    normal_score = model(_normal_images(2)).pred_score
    anomalous_score = model(_anomalous_images(2)).pred_score
    assert normal_score is not None and anomalous_score is not None
    assert torch.all(anomalous_score > normal_score)


# ======================================================================================
# 极简 test runner（不依赖 pytest）
# ======================================================================================
def _collect_tests(keyword: str = "") -> list[tuple[str, Callable[[], Any]]]:
    """收集当前模块里所有 ``test_`` 开头的函数，可按名字关键字过滤。"""
    module = globals()
    return [
        (name, obj)
        for name, obj in sorted(module.items())
        if name.startswith("test_") and callable(obj) and keyword in name
    ]


def main() -> int:
    """运行全部（或按关键字过滤的）测试并打印结果。"""
    import sys

    keyword = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = _collect_tests(keyword)

    print(f"PatchcoreModel 逐函数测试 —— 共 {len(tests)} 项" + (f"（过滤关键字: '{keyword}'）" if keyword else ""))
    print("=" * 78)

    passed, failed = 0, []
    for name, func in tests:
        try:
            func()
        except Exception as err:  # noqa: BLE001
            failed.append((name, err))
            print(f"[FAIL] {name}\n       {type(err).__name__}: {err}")
        else:
            passed += 1
            print(f"[ OK ] {name}")

    print("=" * 78)
    print(f"通过 {passed}/{len(tests)}")
    if failed:
        print("\n失败详情:")
        for name, err in failed:
            print(f"  - {name}: {type(err).__name__}: {err}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
