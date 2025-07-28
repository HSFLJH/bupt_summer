# 该文件提供了 RoIAlign 操作的 Python 接口。
# RoIAlign (Region of Interest Align) 是目标检测和实例分割模型中的一个关键组件，
# 特别是在像 Faster R-CNN 和 Mask R-CNN 这样的两阶段检测器中。
# 它用于从特征图中为每个候选区域（RoI）提取一个固定大小的特征图，以便送入后续的分类和回归头。
#
# 与 RoIPool 不同，RoIAlign 避免了对 RoI 坐标的硬性量化（取整），
# 而是使用双线性插值来计算输入特征图中精确采样点的像素值，从而解决了量化操作带来的不对齐问题。
# 这种对齐对于像素级别的预测任务（如掩码预测）至关重要。

import os
import sys
from typing import List, Union

import torch
import torch.fx
from torchvision.extension import _assert_has_ops
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair
from model.utils.general import check_roi_boxes_shape, convert_boxes_to_roi_format

def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """
    执行 RoIAlign 操作。

    这是一个包装函数，它首先对输入进行一些预处理和检查，然后调用底层的
    `torch.ops.torchvision.roi_align` 实现（通常是 C++ 或 CUDA 加速的）。

    用法:
        # 假设 features 是一个特征图张量
        # proposals 是一个边界框列表
        features = torch.randn(2, 256, 88, 88) # (N, C, H, W)
        proposals = [torch.tensor([[0,0,10,10], [10,10,50,50]], dtype=torch.float32),
                     torch.tensor([[30,30,70,70]], dtype=torch.float32)]
        # 对每个 proposal 提取一个 7x7 的特征图
        pooled_features = roi_align(features, proposals, output_size=(7, 7), spatial_scale=1.0)
        # pooled_features.shape 会是 [3, 256, 7, 7]，因为总共有3个 proposals

    Args:
        input (Tensor): 输入的特征图，形状为 (N, C, H, W)。
        boxes (Union[Tensor, List[Tensor]]): 候选区域（RoIs）。可以是两种格式：
            1. `List[Tensor[K, 4]]`: 每个批次元素的 RoI 列表。
            2. `Tensor[L, 5]`: 所有 RoI 的集合，格式为 `(batch_index, x1, y1, x2, y2)`。
        output_size (Tuple[int, int]): 输出特征图的尺寸 (height, width)。
        spatial_scale (float, optional): 空间尺度因子。用于将 `boxes` 的坐标从输入图像尺度
                                         缩放到特征图尺度。计算方法通常是 `1.0 / feature_stride`。默认为 1.0。
        sampling_ratio (int, optional): 在每个 RoI bin 中用于插值的采样点数量。
                                        如果为 -1，则会自适应地计算（例如，`ceil(roi_height / output_height)`）。默认为 -1。
        aligned (bool, optional): 如果为 True，则使用对齐的坐标变换。这会将 RoI 的角点
                                  移动半个像素，以更好地对齐采样点和输出 bin 的中心。默认为 False。

    Returns:
        Tensor: 从每个 RoI 中提取的池化后的特征图，形状为 `(total_rois, C, output_size_h, output_size_w)`。
    """
    # 检查底层 torchvision C++ 操作是否可用
    _assert_has_ops()
    # 检查输入 boxes 的形状是否合法
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size) # 确保 output_size 是一个 (h, w) 的元组
    # 如果 boxes 是一个列表，则将其转换为 `Tensor[L, 5]` 的格式
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
        
    # 调用底层的 RoIAlign 操作
    return torch.ops.torchvision.roi_align(
        input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
    )
