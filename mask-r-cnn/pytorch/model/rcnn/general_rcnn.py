# 该文件定义了 `GeneralizedRCNN` 类，它是所有 R-CNN 系列模型
# (如 Faster R-CNN, Mask R-CNN) 的一个通用、抽象的基类。
#
# 这个类的核心思想是把一个典型的两阶段检测器分解为四个标准组件：
# 1. `transform`: 数据预处理模块 (在 `model.utils.transforms` 中定义)，负责图像的标准化、
#    尺寸调整和批处理。
# 2. `backbone`: 特征提取网络 (如 ResNet+FPN)，负责从输入图像中提取多尺度特征图。
# 3. `rpn` (Region Proposal Network): 区域提议网络，负责从特征图上生成候选区域 (proposals)。
# 4. `roi_heads` (Region of Interest Heads): RoI 头部，负责对 RPN 提出的候选区域进行
#    精细的分类、边界框回归以及（对于 Mask R-CNN）掩码预测。
#
# 通过将这些组件组合在一起，`GeneralizedRCNN` 定义了从输入图像到最终检测结果的完整前向传播逻辑。

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor


class GeneralizedRCNN(nn.Module):
    """
    所有 R-CNN 模型的通用基类。

    Args:
        backbone (nn.Module): 特征提取骨干网络。
        rpn (nn.Module): 区域提议网络。
        roi_heads (nn.Module): RoI 头部 (用于分类、回归、掩码预测)。
        transform (nn.Module): 预处理和后处理模块。
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # 仅在 torchscript 模式下使用
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """根据模型是否处于训练模式，决定返回损失还是检测结果。"""
        if self.training:
            return losses

        return detections

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        `GeneralizedRCNN` 的主前向传播逻辑。

        Args:
            images (List[Tensor]): 输入的图像列表，每张图像是一个 `[C, H, W]` 的张量。
            targets (Optional[List[Dict[str, Tensor]]]): 真实标注的列表，仅在训练时提供。

        Returns:
            Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
                - 在训练模式下，返回一个包含各项损失 (如 `loss_objectness`, `loss_rpn_box_reg`,
                  `loss_classifier`, `loss_box_reg`, `loss_mask`) 的字典。
                - 在评估模式下，返回一个包含检测结果 (如 `boxes`, `labels`, `scores`, `masks`) 的列表。
        """
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None when in training mode")
            # 检查 targets 中 boxes 的格式是否正确
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise TypeError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        # 记录原始图像尺寸，用于后续的后处理
        original_image_sizes: List[Tuple[int, int]] = [img.shape[-2:] for img in images]

        # 1. 预处理：对图像和标注进行标准化和尺寸调整
        images, targets = self.transform(images, targets)

        # 检查是否存在退化的边界框 (宽或高为0)
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # 打印出第一个退化的边界框信息
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        # 2. Backbone: 提取特征
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        
        # 3. RPN: 生成候选区域
        # proposals 是候选框列表; proposal_losses 是 RPN 的损失 (仅训练时)
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        # 4. RoI Heads: 对候选区域进行分类、回归和掩码预测
        # detections 是最终的检测结果; detector_losses 是 RoI 头的损失 (仅训练时)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        # 5. 后处理：将检测结果从推理尺寸映射回原始图像尺寸
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # 将 RPN 和 RoI Head 的损失合并
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        
        return self.eager_outputs(losses, detections)
