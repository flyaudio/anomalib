import math
import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor


def test_feature_extractor():
    '''
    Key Features

Symbolic Tracing: The function uses torch.fx to trace the computation graph and extract specified nodes.

Custom Node Names: You can rename extracted nodes for clarity.

Flexible Node Selection: Specify nodes using full paths (e.g., layer4.2.relu) or truncated names (e.g., layer4).

Important Considerations

Node Naming: Use get_graph_node_names(model) to list all available nodes for extraction.

Leaf Modules: If certain submodules are problematic during tracing, mark them as "leaf modules" using tracer_kwargs.

Memory Usage: Extracting multiple layers can increase memory consumption.

This utility simplifies accessing intermediate representations in deep learning models for various computer vision tasks.
    '''
    # Load a pre-trained ResNet18 model
    model = resnet18(pretrained=True)
    # Specify the layers to extract features from
    return_nodes = {
    'layer1': 'feat1', # Rename layer1 output to 'feat1'
    'layer3': 'feat2' # Rename layer3 output to 'feat2'
    }
    # Create the feature extractor
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    # Pass a sample input through the feature extractor
    input_tensor = torch.rand(1, 3, 224, 224)
    output = feature_extractor(input_tensor)
    # Print the extracted features and their shapes
    for name, feature in output.items():
        print(f"{name}: {feature.shape}")




def resize_pos_embed(posemb, # [1, 197, 768]：1 个 batch，197 个位置（1 个 cls + 196 个 patch），每个位置是 768 维向量
                     posemb_new, 
                     num_tokens=1, # 前面有几个非 patch 的 token（ViT 是 1 个 cls）
                     gs_new=() # 新网格尺寸，如 (16, 16)
                     ):
    ntok_new = posemb_new.shape[1]          # 新模型需要的位置总数，比如 257
    if num_tokens:
        # ① 把 cls token 和 patch 位置拆开
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens              # 新的 patch 数量：257-1 = 256
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # ② 由 token 数反推旧网格边长：sqrt(196) = 14
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2   # 新网格：[16, 16]

    # ③ 拉回方阵 → 变成图片张量格式 → 插值放大 → 再拉平
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0]*gs_new[1], -1)

    # ④ cls token 原样拼回最前面
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb