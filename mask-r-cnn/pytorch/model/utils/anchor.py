# 该文件负责生成目标检测模型中所需的锚框 (Anchor Boxes) 或默认框 (Default Boxes)。
# 锚框是模型在特征图上预设的一系列参考框，用于预测物体的偏移量和类别。
# 这个文件主要包含两种生成器：
# 1. AnchorGenerator: 用于 Faster R-CNN / Mask R-CNN 系列模型。
# 2. DefaultBoxGenerator: 用于 SSD 系列模型。

import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor


class ImageList:
    """
    一个简单的数据结构，用于将一个批次中可能具有不同尺寸的多个图像存储在一个单一的张量中。
    它通过将所有图像填充（padding）到该批次中最大图像的尺寸来实现这一点，
    并同时保存每个图像的原始尺寸。

    这在处理批处理数据时非常方便，因为标准的 PyTorch 张量要求同一批次中的所有元素具有相同的尺寸。

    Args:
        tensors (Tensor): 一个包含了填充后图像的张量，形状通常是 `[N, C, H, W]`，
                          其中 H 和 W 是批次中最大的高度和宽度。
        image_sizes (List[Tuple[int, int]]): 一个列表，包含了该批次中每个图像的原始尺寸 `(height, width)`。
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        """
        将内部的张量移动到指定的设备 (CPU or GPU)。

        Args:
            device (torch.device): 目标设备。

        Returns:
            ImageList: 一个新的、张量在指定设备上的 `ImageList` 实例。
        """
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class AnchorGenerator(nn.Module):
    """
    为一组特征图生成锚框的模块。这是 Faster R-CNN 和 Mask R-CNN 中 RPN 的核心组件。
    该模块支持在每个特征图的每个空间位置上，计算具有多种尺寸和多种长宽比的锚框。

    `sizes` 和 `aspect_ratios` 参数应该是长度相同的元组，其长度应与输入的特征图数量相对应。
    `sizes[i]` 和 `aspect_ratios[i]` 分别定义了第 `i` 个特征图上使用的锚框尺寸和长宽比。

    Args:
        sizes (Tuple[Tuple[int]]): 一个元组的元组，定义了每个特征图层级上使用的基础锚框尺寸。
                                   例如 `((32,), (64,), (128,), (256,), (512,))`。
        aspect_ratios (Tuple[Tuple[float]]): 一个元组的元组，定义了每个特征图层级上使用的长宽比。
                                             例如 `((0.5, 1.0, 2.0),)` 会被应用到所有层级。
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super().__init__()
        # 确保 sizes 和 aspect_ratios 具有正确的嵌套结构
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
        # `cell_anchors` 是一个列表，存储了每个特征图层级的基础锚框模板。
        # 这些是中心点在 (0, 0) 的锚框，后续将被平移到特征图的每个位置。
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """
        为给定的尺度和长宽比组合生成一组以 (0,0) 为中心的锚框模板。

        Args:
            scales (List[int]): 锚框的基础尺寸（面积的平方根）。
            aspect_ratios (List[float]): 锚框的长宽比 (height / width)。
            dtype (torch.dtype): 输出张量的数据类型。
            device (torch.device): 输出张量的设备。

        Returns:
            Tensor: 生成的锚框模板，形状为 `[num_scales * num_ratios, 4]`，格式为 `(x1, y1, x2, y2)`。
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        # 计算所有组合的宽度和高度
        # ws 和 hs 的形状都是 [num_ratios * num_scales]
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # 创建以 (0,0) 为中心的锚框，格式为 (x1, y1, x2, y2)
        # base_anchors 的形状是 [num_ratios * num_scales, 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        """一个辅助函数，用于将 `cell_anchors` 移动到正确的设备和数据类型。"""
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self) -> List[int]:
        """返回每个特征图层级上，每个空间位置生成的锚框数量。"""
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        """
        将锚框模板（cell_anchors）平铺到整个特征图网格上。

        Args:
            grid_sizes (List[List[int]]): 每个特征图的尺寸 `(height, width)` 列表。
            strides (List[List[Tensor]]): 每个特征图相对于输入图像的步长 `(stride_y, stride_x)` 列表。

        Returns:
            List[Tensor]: 一个列表，每个元素是对应特征图上生成的所有锚框，形状为 `[grid_h * grid_w * num_anchors, 4]`。
        """
        anchors = []
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "特征图、步长和 cell_anchors 的数量必须匹配。",
        )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # 生成网格中每个点的中心坐标
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            # 使用 torch.meshgrid 生成网格坐标
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            # shifts 的形状是 [grid_h * grid_w, 4]，内容是 (x_center, y_center, x_center, y_center)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # 通过广播机制，将中心在 (0,0) 的 base_anchors 添加到每个网格点的中心坐标上
            # (shifts.view(-1, 1, 4) 的形状是 [grid_h*grid_w, 1, 4])
            # (base_anchors.view(1, -1, 4) 的形状是 [1, num_anchors, 4])
            # 最终结果的形状是 [grid_h*grid_w, num_anchors, 4]，然后 reshape 成 [N, 4]
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        """
        模块的主前向传播函数。

        Args:
            image_list (ImageList): 包含了输入图像批次和原始尺寸。
            feature_maps (List[Tensor]): 从 FPN 输出的多层级特征图列表。

        Returns:
            List[Tensor]: 一个列表，长度为批次大小。每个元素是对应图像上生成的所有锚框的张量。
        """
        # 获取每个特征图的尺寸
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        # 获取填充后的大图像的尺寸
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        
        # 计算每个特征图相对于输入图像的步长 (stride)
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        
        # 将 cell_anchors 移动到正确的设备
        self.set_cell_anchors(dtype, device)
        # 在所有特征图上生成锚框
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        
        anchors: List[List[torch.Tensor]] = []
        # 为批次中的每张图片生成一个锚框列表
        # 注意：这里生成的锚框是针对整个填充后的大图像的，并且对于批次中的每张图片都是相同的。
        # 后续的处理（如筛选、裁剪）会根据每张图片的实际尺寸进行。
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        # 将每个图像的所有特征图的锚框拼接成一个张量
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


class DefaultBoxGenerator(nn.Module):
    """
    为 SSD (Single Shot MultiBox Detector) 模型生成默认框（Default Boxes）的模块。
    其生成逻辑与 Faster R-CNN 的 AnchorGenerator 有所不同，特别是在尺度的计算和长宽比的处理上。

    Args:
        aspect_ratios (List[List[int]]): 每个特征图上使用的长宽比列表。
        min_ratio (float): 用于估算尺度的最小比例 s_min。
        max_ratio (float): 用于估算尺度的最大比例 s_max。
        scales (List[float], optional): 可选，直接提供每个层级的默认框尺度。如果为 None，则根据 min/max_ratio 自动计算。
        steps (List[int], optional): 可选，每个特征图的步长。如果为 None，则根据特征图尺寸自动推断。
        clip (bool): 是否将默认框的坐标裁剪到 [0, 1] 范围内。
    """

    def __init__(
        self,
        aspect_ratios: List[List[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        clip: bool = True,
    ):
        super().__init__()
        if steps is not None and len(aspect_ratios) != len(steps):
            raise ValueError("aspect_ratios 和 steps 的长度必须相同")
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip
        num_outputs = len(aspect_ratios)

        # 根据 SSD 论文估算默认框的尺度
        if scales is None:
            if num_outputs > 1:
                range_ratio = max_ratio - min_ratio
                # 线性插值计算每个层级的尺度 s_k
                self.scales = [min_ratio + range_ratio * k / (num_outputs - 1.0) for k in range(num_outputs)]
                self.scales.append(1.0) # 额外添加一个尺度
            else:
                self.scales = [min_ratio, max_ratio]
        else:
            self.scales = scales

        # 生成所有层级的 [width, height] 组合模板
        self._wh_pairs = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(
        self, num_outputs: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
    ) -> List[Tensor]:
        """为每个特征图层级生成 [width, height] 模板对。"""
        _wh_pairs: List[Tensor] = []
        for k in range(num_outputs):
            # 对于每个层级 k，首先添加两个特殊的默认框：
            # 1. 长宽比为 1，尺度为 s_k
            # 2. 长宽比为 1，尺度为 s'_k = sqrt(s_k * s_{k+1})
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # 然后为该层级的其他长宽比 ar 添加框
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]]) # 同时添加 (w,h) 和 (h,w)

            _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
        return _wh_pairs

    def num_anchors_per_location(self) -> List[int]:
        """返回每个特征图层级上，每个空间位置生成的默认框数量。"""
        # 2 个基础框 + 2 * (其他长宽比的数量)
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    def _grid_default_boxes(
        self, grid_sizes: List[List[int]], image_size: List[int], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """
        在所有特征图网格上生成默认框。
        返回的框是 (cx, cy, w, h) 格式，并相对于图像尺寸进行了归一化。
        """
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # 计算网格中心点坐标
            if self.steps is not None:
                # 使用预设的步长
                x_f_k, y_f_k = self.steps[k], self.steps[k]
            else:
                # 根据特征图尺寸自动推断步长
                y_f_k, x_f_k = image_size[0] / f_k[0], image_size[1] / f_k[1]

            shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype=dtype)
            shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype=dtype)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            
            # shifts 的形状是 [num_cells * 2, 2]，其中 num_cells = f_k[0] * f_k[1]
            shifts = torch.stack((shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1).reshape(-1, 2)
            
            # 如果需要，将 w, h 裁剪到 [0, 1]
            _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
            # 将 [w, h] 模板复制到每个网格位置
            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)
            
            # 拼接中心点和宽高，得到 (cx, cy, w, h) 格式的默认框
            default_box = torch.cat((shifts, wh_pairs), dim=1)
            default_boxes.append(default_box)

        return torch.cat(default_boxes, dim=0)

    def __repr__(self) -> str:
        """自定义模块的打印输出信息。"""
        s = (
            f"{self.__class__.__name__}("
            f"aspect_ratios={self.aspect_ratios}"
            f", clip={self.clip}"
            f", scales={self.scales}"
            f", steps={self.steps}"
            ")"
        )
        return s

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        """
        模块的主前向传播函数。

        Args:
            image_list (ImageList): 包含了输入图像批次和原始尺寸。
            feature_maps (List[Tensor]): 多层级特征图列表。

        Returns:
            List[Tensor]: 一个列表，每个元素是对应图像上生成的所有默认框的张量，
                          格式为 `(x1, y1, x2, y2)`，且坐标是相对于原始图像尺寸的。
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        
        # 生成归一化的 (cx, cy, w, h) 格式的默认框
        default_boxes = self._grid_default_boxes(grid_sizes, list(image_size), dtype=dtype)
        default_boxes = default_boxes.to(device)

        dboxes = []
        # 将归一化的坐标转换为相对于原始图像尺寸的 (x1, y1, x2, y2) 格式
        # 注意：这里对于批次中的所有图片，都使用了相同的原始图像尺寸进行转换，
        # 这可能是一个简化处理，或者假设所有输入图像在送入模型前已被缩放到相同尺寸。
        x_y_size = torch.tensor([image_size[1], image_size[0]], device=default_boxes.device)
        for _ in image_list.image_sizes:
            dboxes_in_image = default_boxes
            # (cx, cy, w, h) -> (x1, y1, x2, y2)
            dboxes_in_image = torch.cat(
                [
                    (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                    (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                ],
                -1,
            )
            dboxes.append(dboxes_in_image)
        return dboxes
