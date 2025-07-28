# 该文件实现了用于边界框回归的几种高级 IoU-based 损失函数，
# 包括 DIoU (Distance-IoU) 和 CIoU (Complete-IoU) loss。
# 这些损失函数在传统 IoU loss 的基础上，考虑了边界框中心点的距离、
# 重叠面积以及长宽比等因素，能够提供更稳定和准确的回归梯度，
# 从而在训练中实现更快的收敛和更高的定位精度。

import torch
from torch import Tensor
from typing import Tuple

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def _upcast_non_float(t: Tensor) -> Tensor:
    """
    一个内部辅助函数，用于将非浮点类型的张量提升为浮点类型。
    这可以防止在后续的乘法等运算中因整数类型导致数值溢出。

    Args:
        t (Tensor): 输入张量。

    Returns:
        Tensor: 如果需要，则返回 upcast 后的浮点张量。
    """
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.dtype not in (torch.float32, torch.float64):
        return t.float()
    return t


def distance_box_iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    计算 Distance-IoU (DIoU) 损失。
    DIoU Loss = 1 - IoU + (中心点距离惩罚项)
    它在 IoU 损失的基础上，直接最小化两个框中心点之间的归一化距离，
    从而比 GIoU Loss 收敛更快，特别是在两个框不重叠时。

    Args:
        boxes1 (torch.Tensor): 预测的边界框，形状为 `[N, 4]`。
        boxes2 (torch.Tensor): 真实的边界框，形状为 `[N, 4]`。
        reduction (str, optional): 指定应用于输出的归约方式：
            'none' | 'mean' | 'sum'。默认为 'none'。
        eps (float, optional): 一个小的数值，用于防止除以零。默认为 1e-7。

    Returns:
        torch.Tensor: 计算出的 DIoU 损失。
    """
    # Original Implementation from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/losses.py

    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)

    loss, _ = _diou_iou_loss(boxes1, boxes2, eps)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def _loss_inter_union(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """内部函数，计算两个边界框集合的交集和并集面积。"""
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

    return intsctk, unionk


def _diou_iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """内部函数，计算 DIoU 损失的核心部分和 IoU 值。"""
    intsct, union = _loss_inter_union(boxes1, boxes2)
    iou = intsct / (union + eps)

    # 计算能同时包含两个框的最小闭包框 (smallest enclosing box)
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    # 最小闭包框的对角线距离的平方
    diagonal_distance_squared = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps
    
    # 两个框中心点的坐标
    x_p = (x1 + x2) / 2
    y_p = (y1 + y2) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    
    # 两个框中心点距离的平方
    centers_distance_squared = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    
    # DIoU 损失 = 1 - IoU + (中心点距离惩罚项)
    loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared)
    return loss, iou


def complete_box_iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    计算 Complete-IoU (CIoU) 损失。
    CIoU Loss = 1 - IoU + (中心点距离惩罚项) + (长宽比一致性惩罚项)
    它在 DIoU 的基础上，额外增加了一个惩罚项来考虑预测框和真实框之间长宽比的一致性。
    这使得回归过程更加稳定，避免模型在训练早期为了快速减小中心点距离而"胡乱"放大预测框。

    Args:
        boxes1 (torch.Tensor): 预测的边界框，形状为 `[N, 4]`。
        boxes2 (torch.Tensor): 真实的边界框，形状为 `[N, 4]`。
        reduction (str, optional): 指定应用于输出的归约方式：
            'none' | 'mean' | 'sum'。默认为 'none'。
        eps (float, optional): 一个小的数值，用于防止除以零。默认为 1e-7。

    Returns:
        torch.Tensor: 计算出的 CIoU 损失。
    """
    # Original Implementation from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/losses.py


    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)

    # 首先计算 DIoU 损失和 IoU
    diou_loss, iou = _diou_iou_loss(boxes1, boxes2, eps)

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # 计算两个框的宽度和高度
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    
    # 计算长宽比一致性惩罚项 v
    # v = (4 / π^2) * (arctan(w_gt / h_gt) - arctan(w_pred / h_pred))^2
    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred), 2)
    
    # 计算权重因子 alpha，用于平衡惩罚项
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU 损失 = DIoU 损失 + alpha * v
    loss = diou_loss + alpha * v
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

