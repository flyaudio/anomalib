"""PatchCore 特征 embedding 的交互式 3D 可视化(基于 Rerun)。

功能:
- 扫描指定文件夹的全部图片,用 PatchCore 骨干提取特征,全局平均池化拼接后 PCA 降到 3D
- 3D 空间鼠标旋转(左键拖拽)/ 平移(右键拖拽)/ 缩放(滚轮)
- 悬停或点击 3D 点 → 显示对应图片;点击图片 → 高亮对应点
- 框选/套索 3D 区域 → 选中点的图片集中显示在 Selection 面板

用法:
    python embedding_viewer.py --folder datasets/xx --backbone wide_resnet50_2
    python embedding_viewer.py --folder ~/.cache/anomalib/datasets/MVTecAD/pill/test --backbone yolo11n

默认会打开浏览器(http://127.0.0.1:9876),同时保存 embeddings.rrd 可离线回放。
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.v2.functional import resize

import rerun as rr
from anomalib.models.components.feature_extractors import TimmFeatureExtractor

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def collect_images(folder: str, max_images: int) -> tuple[list[Path], list[str], list[int]]:
    """递归收集图片,label = 图片相对 folder 的父目录路径。"""
    root = Path(folder).resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)[:max_images]
    labels = []
    for p in paths:
        rel = p.relative_to(root).parent
        labels.append(rel.as_posix() if rel.as_posix() != "." else root.name)
    uniq = sorted(set(labels))
    label_idx = [uniq.index(l) for l in labels]
    return paths, labels, label_idx


def build_extractor(backbone: str) -> tuple[TimmFeatureExtractor, bool]:
    """构建特征提取器,返回 (extractor, 是否 yolo 输入模式)。"""
    if backbone == "yolo11n":
        from ultralytics import YOLO

        from yolo_backbone_patchcore import YoloBackbone, patch_c2f_for_fx

        patch_c2f_for_fx()
        seq = YOLO("yolo11n.pt").model.model[:7]
        extractor = TimmFeatureExtractor(backbone=YoloBackbone(seq), layers=["seq.4", "seq.6"])
        return extractor.cuda().eval(), True
    extractor = TimmFeatureExtractor(backbone="wide_resnet50_2", layers=["layer2", "layer3"])
    return extractor.cuda().eval(), False


def compute_embeddings(
    extractor: TimmFeatureExtractor,
    paths: list[Path],
    yolo_input: bool,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """逐图提取特征:各层 GAP 后拼接。返回 (N,D) embedding 和原图 uint8 数组。"""
    embs: list[torch.Tensor] = []
    images: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = read_image(str(p)).float() / 255.0
            img = resize(img, [256, 256], antialias=True)
            images.append((img * 255).permute(1, 2, 0).byte().numpy())
            imgs.append(img)
        x = torch.stack(imgs).cuda()
        if not yolo_input:
            x = (x - IMAGENET_MEAN.cuda()) / IMAGENET_STD.cuda()
        with torch.no_grad():
            feats = extractor(x)  # dict: layer -> (B,C,H,W)
            emb = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats.values()], 1)
        embs.append(emb.cpu())
    return torch.cat(embs).numpy(), images


def pca3(x: np.ndarray) -> np.ndarray:
    """PCA 降到 3 维(SVD,无额外依赖)。"""
    x = torch.tensor(x, dtype=torch.float32)
    x = x - x.mean(0)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    return (x @ v[:3].T).numpy()


def label_color(i: int, n: int) -> tuple[int, int, int]:
    """给每个 label 一个高饱和度的区分色。"""
    r, g, b = colorsys.hsv_to_rgb((i % n) / max(n, 1), 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore embedding 3D 可视化")
    parser.add_argument("--folder", type=str, required=True, help="图片文件夹(递归扫描)")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", choices=["wide_resnet50_2", "yolo11n"])
    parser.add_argument("--max-images", type=int, default=2000, help="最多加载的图片数")
    parser.add_argument("--no-viewer", action="store_true", help="只保存 .rrd 不打开浏览器(用于验证)")
    args = parser.parse_args()

    paths, labels, label_idx = collect_images(args.folder, args.max_images)
    uniq_labels = sorted(set(labels))
    if not paths:
        raise SystemExit(f"文件夹里没有图片: {args.folder}")
    print(f"图片 {len(paths)} 张,label 类别 {len(uniq_labels)} 个: {uniq_labels}")

    extractor, yolo_input = build_extractor(args.backbone)
    print("提取 embedding ...")
    embeddings, images = compute_embeddings(extractor, paths, yolo_input)
    print(f"embedding 维度: {embeddings.shape}")

    print("PCA 降维到 3D ...")
    xyz = pca3(embeddings)

    rr.init("patchcore-embeddings", spawn=not args.no_viewer)

    # 点和图注册到同一实体路径 → 悬停/选中双向联动
    colors = [label_color(label_idx[i], len(uniq_labels)) for i in range(len(paths))]
    for i, p in enumerate(paths):
        entity = f"embeddings/{labels[i]}/{p.stem}"
        rr.log(
            entity,
            rr.Points3D(
                positions=xyz[i : i + 1],
                colors=np.array([colors[i]], dtype=np.uint8),
                radii=0.02,
            ),
            static=True,
        )
        rr.log(entity, rr.Image(images[i]), static=True)

    print("\n图例(颜色 → label):")
    for i, lab in enumerate(uniq_labels):
        c = label_color(i, len(uniq_labels))
        print(f"  RGB{c}  {lab}")

    rr.save("embeddings.rrd")
    print("\n已保存 embeddings.rrd")
    if not args.no_viewer:
        print("打开浏览器查看(左键旋转/右键平移/滚轮缩放;悬停点看图;框选后看 Selection 面板)。Ctrl+C 退出。")
        rr.serve()


if __name__ == "__main__":
    main()
