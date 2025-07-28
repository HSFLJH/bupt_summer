# 该文件实现了 Multi-Scale RoI Align (多尺度兴趣区域对齐) 操作。
# 这是现代两阶段目标检测器（如 Faster R-CNN, Mask R-CNN）中的一个核心组件，
# 它将 FPN (特征金字塔网络) 的思想与 RoIAlign 操作结合起来。
#
# 核心思想是：对于每一个候选区域 (RoI)，不再是从单一的特征图上提取特征，
# 而是根据 RoI 的大小，动态地为其分配合适的 FPN 特征层级。
# 具体来说，大的 RoI 应该从分辨率较低但语义信息更强的 FPN 高层特征图中提取特征，
# 而小的 RoI 则应该从分辨率较高但语义信息较弱的 FPN 低层特征图中提取。
#
# 这种机制使得模型能够更好地处理不同尺度的物体。

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from model.utils.boxes import box_area
from model.utils.roi_align import roi_align


class LevelMapper:
    """
    一个用于将每个 RoI 映射到其最合适的 FPN 特征层级的计算模块。
    它实现了 FPN 论文中提出的启发式规则（公式 1）。

    计算公式为:
        level = floor(level0 + log2(sqrt(area) / scale0))

    其中 `area` 是 RoI 的面积，`scale0` 是基准尺寸（通常是224），`level0` 是基准尺寸对应的层级（通常是4，对应ResNet的C4）。

    Args:
        k_min (int): FPN 的最低层级索引。
        k_max (int): FPN 的最高层级索引。
        canonical_scale (int): RoI 的基准尺寸，即论文中的 224。
        canonical_level (int): 基准尺寸对应的 FPN 层级，即论文中的 k0=4。
        eps (float): 一个很小的数，用于防止 log2(0) 的计算错误。
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        计算一批 RoI 应该被分配到的 FPN 层级。

        Args:
            boxlists (List[Tensor]): 一个边界框列表，每个元素是代表一张图片上所有 RoI 的张量。

        Returns:
            Tensor: 一个张量，包含了每个 RoI 对应的目标层级索引。
        """
        # 计算所有 RoI 的面积，然后取平方根，得到近似的边长 s
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # 应用 FPN 论文中的公式 (1)
        # torch.log2(s / self.s0) 计算了 RoI 尺寸相对于基准尺寸的对数尺度变化
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        
        # 将计算出的层级限制在 FPN 的有效层级范围 [k_min, k_max] 内
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        
        # 将绝对层级（如 2,3,4,5）转换为相对索引（如 0,1,2,3）
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
    eps: float = 1e-6,
) -> LevelMapper:
    """LevelMapper 的工厂函数，方便创建实例。"""
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


def _infer_scale(feature: Tensor, original_size: List[int]) -> float:
    """
    一个内部函数，用于推断单个特征图相对于原始输入图像的空间缩放比例。
    它假设缩放比例是 2 的整数次幂。
    """
    size = feature.shape[-2:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        # 通过 log2 -> round -> pow(2) 来找到最接近的 2^k 形式的缩放比例
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    # 通常 H 和 W 方向的缩放比例是相同的
    return possible_scales[0]


def _convert_to_roi_format(boxes: List[Tensor]) -> Tensor:
    """
    将 List[Tensor[N, 4]] 格式的 boxes 转换为 Tensor[K, 5] 的 RoI 格式 (batch_idx, x1, y1, x2, y2)。
    这是一个在 `general.py` 中 `convert_boxes_to_roi_format` 的重复实现，可能为了内部解耦。
    """
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def _setup_scales(
    features: List[Tensor], image_shapes: List[Tuple[int, int]], canonical_scale: int, canonical_level: int
) -> Tuple[List[float], LevelMapper]:
    """
    一个设置函数，用于计算所有 FPN 层级的空间缩放比例并初始化 LevelMapper。
    这个函数只在第一次前向传播时被调用。
    """
    if not image_shapes:
        raise ValueError("images list should not be empty")
        
    # 找到批次中最大的图像尺寸作为参考
    max_x = 0
    max_y = 0
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
    original_input_shape = (max_x, max_y)

    # 推断每个 FPN 特征层级的缩放比例
    scales = [_infer_scale(feat, original_input_shape) for feat in features]
    
    # 利用缩放比例是 2 的整数次幂这一事实，计算 FPN 的最小和最大层级
    # level = -log2(scale)
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

    # 初始化 LevelMapper
    map_levels = initLevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return scales, map_levels


def _filter_input(x: Dict[str, Tensor], featmap_names: List[str]) -> List[Tensor]:
    """根据 `featmap_names` 从输入的特征图字典中过滤出需要的特征图列表。"""
    x_filtered = []
    for k, v in x.items():
        if k in featmap_names:
            x_filtered.append(v)
    return x_filtered


def _onnx_merge_levels(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
    """一个专门为 ONNX 导出设计的函数，用于将不同层级的 RoIAlign 结果合并回一个张量。"""
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    # 创建一个零张量用于存放最终结果
    res = torch.zeros(
        (levels.size(0), first_result.size(1), first_result.size(2), first_result.size(3)), dtype=dtype, device=device
    )
    # 遍历每个层级
    for level, per_level_result in enumerate(unmerged_results):
        # 找到属于当前层级的 RoI 的索引
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        # 将索引扩展到与结果张量相同的维度
        index = index.expand(
            index.size(0),
            per_level_result.size(1),
            per_level_result.size(2),
            per_level_result.size(3),
        )
        # 使用 scatter 将当前层级的结果填充到 `res` 张量的对应位置
        res = res.scatter(0, index, per_level_result)
    return res


def _multiscale_roi_align(
    x_filtered: List[Tensor],
    boxes: List[Tensor],
    output_size: Tuple[int, int],
    sampling_ratio: int,
    scales: List[float],
    mapper: LevelMapper,
) -> Tensor:
    """
    执行多尺度 RoI Align 的核心逻辑。

    Args:
        x_filtered (List[Tensor]): 输入的 FPN 特征图列表。
        boxes (List[Tensor[N, 4]]): RoI 列表。
        output_size (Tuple[int, int]): RoIAlign 的输出尺寸。
        sampling_ratio (int): RoIAlign 的采样率。
        scales (List[float]): 每个 FPN 层级的空间缩放比例。
        mapper (LevelMapper): 用于分配 RoI 到 FPN 层级的 LevelMapper 实例。

    Returns:
        Tensor: 所有 RoI 池化后的特征，按原始顺序排列。
    """

    num_levels = len(x_filtered)
    # 将 RoI 列表转换为 [batch_idx, x1, y1, x2, y2] 格式
    rois = _convert_to_roi_format(boxes)

    # 如果只有一个特征层级，则退化为标准的 RoIAlign
    if num_levels == 1:
        return roi_align(
            x_filtered[0], rois, output_size=output_size, spatial_scale=scales[0], sampling_ratio=sampling_ratio
        )

    # 使用 mapper 计算每个 RoI 对应的层级索引
    levels = mapper(boxes)

    num_rois = len(rois)
    num_channels = x_filtered[0].shape[1]
    dtype, device = x_filtered[0].dtype, x_filtered[0].device
    
    # 创建一个空的张量来存储最终结果
    result = torch.zeros((num_rois, num_channels) + output_size, dtype=dtype, device=device)

    # 用于 ONNX 追踪的结果列表
    tracing_results = []
    # 遍历每个 FPN 层级
    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        # 找到被分配到当前层级的所有 RoI 的索引
        idx_in_level = torch.where(levels == level)[0]
        # 获取这些 RoI
        rois_per_level = rois[idx_in_level]

        # 对这些 RoI 执行 RoIAlign
        result_idx_in_level = roi_align(
            per_level_feature, rois_per_level, output_size=output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
        )

        if torchvision._is_tracing():
            tracing_results.append(result_idx_in_level.to(dtype))
        else:
            # 使用计算出的索引，将当前层级的结果放回 `result` 张量的正确位置
            # 注意：这里需要手动转换数据类型以匹配 `result` 张量，以处理自动混合精度（AMP）的场景。
            result[idx_in_level] = result_idx_in_level.to(result.dtype)

    # 如果正在进行 ONNX 追踪，则使用特殊的合并函数
    if torchvision._is_tracing():
        result = _onnx_merge_levels(levels, tracing_results)

    return result


class MultiScaleRoIAlign(nn.Module):
    """
    多尺度 RoI Align 模块的高层封装。
    这是一个 `nn.Module`，可以方便地集成到更大的模型中。

    它在内部维护了 `scales` 和 `LevelMapper`，并在第一次前向传播时自动进行初始化。

    Args:
        featmap_names (List[str]): 一个列表，包含了要从输入字典中使用的 FPN 特征图的名称。
                                   例如：['p2', 'p3', 'p4', 'p5']。
        output_size (Union[int, Tuple[int]]): RoIAlign 的输出尺寸。
        sampling_ratio (int): RoIAlign 的采样率。
        canonical_scale (int, optional): 传递给 LevelMapper 的基准尺寸。
        canonical_level (int, optional): 传递给 LevelMapper 的基准层级。
    """
    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        
        # `scales` 和 `map_levels` 初始化为 None，它们将在第一次 forward 调用时被计算和设置。
        self.scales = None
        self.map_levels = None
        
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        执行多尺度 RoI Align 的前向传播。

        Args:
            x (Dict[str, Tensor]): 来自 FPN 的特征图字典。
            boxes (List[Tensor]): 每个图像的 RoI 列表。
            image_shapes (List[Tuple[int, int]]): 每个图像的原始尺寸 (height, width) 的列表。

        Returns:
            Tensor: 所有 RoI 池化后的特征。
        """
        # 从输入字典中过滤出需要的特征图
        x_filtered = _filter_input(x, self.featmap_names)
        
        # 如果 scales 和 map_levels 尚未被设置（即第一次调用），则进行设置
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )
        
        # 调用核心的执行函数
        return _multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )

    def __repr__(self) -> str:
        """自定义模块的打印输出信息。"""
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )
