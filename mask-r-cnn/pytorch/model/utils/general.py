# 该文件包含了一些通用的、不便于归类到其他特定文件中的神经网络工具和辅助函数。
# 这些工具在构建复杂的模型结构（如带有FPN的骨干网络、处理边界框和掩码）时非常有用。
# 主要包括：
# - IntermediateLayerGetter: 用于从模型中提取中间层特征，是构建FPN的关键。
# - Box/RoI处理函数: 如格式转换、形状检查等。
# - Mask处理函数: 如扩展、粘贴等，用于Mask R-CNN。
# - 训练辅助工具: 如BalancedPositiveNegativeSampler（正负样本采样器）、Matcher（匹配器）和BoxCoder（边界框编解码器）。

from typing import Any, Dict, List, Optional, Tuple, Union
import collections
from collections import OrderedDict
from itertools import repeat
from torch import nn, Tensor
import torch
import math
import torch.nn.functional as F
import torchvision


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    一个辅助函数，用于将输入 `x` 转换为一个长度为 `n` 的元组。
    这在处理卷积层的 `kernel_size`, `stride`, `padding` 等参数时非常有用，
    因为这些参数既可以是一个整数，也可以是一个元组。

    用法示例:
        _make_ntuple(3, 2)         # -> (3, 3)
        _make_ntuple((3, 4), 2)    # -> (3, 4)
        _make_ntuple([3, 4], 2)    # -> (3, 4)

    Args:
        x (Any): 输入值。可以是一个可迭代对象（如列表、元组）或单个值（如整数）。
        n (int): 目标元组的长度。

    Returns:
        Tuple[Any, ...]: 一个长度为 n 的元组。
    """
    # 如果 x 已经是可迭代对象（但不是字符串），则直接将其转换为元组返回
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    # 如果 x 是单个值，则使用 itertools.repeat 将其重复 n 次，并构造成元组
    return tuple(repeat(x, n))


class IntermediateLayerGetter(nn.ModuleDict):
    """
    一个非常重要的工具类，用于从一个模型中提取并返回其一个或多个中间层的输出。
    它通过 "hook" 的思想，在不修改原始模型代码的情况下，捕获模型在前向传播过程中的中间结果。
    这对于构建特征金字塔网络 (FPN) 等需要多尺度特征的结构至关重要。

    它继承自 `nn.ModuleDict`，这意味着它本身也是一个 PyTorch 模块。

    Args:
        model (nn.Module): 需要从中提取特征的原始模型（例如一个 ResNet 实例）。
        return_layers (Dict[str, str]): 一个字典，指定了要返回的层和它们的新名称。
            - key: 原始模型中模块的名称 (必须是 `model.named_children()` 中存在的)。
            - value: 在输出字典中为该层特征图赋予的新名称。
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        # 检查 `return_layers` 中请求的层是否存在于模型中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        
        # 创建一个新的 OrderedDict 来存储需要执行前向传播的层
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            # 如果某个层是需要返回的层，我们只是记录它，但不会停止构建网络
            # 直到所有需要的层都被包含进来后，才停止。
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        
        # 调用父类构造函数，将需要执行的层作为子模块
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        """
        执行前向传播并捕获指定层的输出。

        Args:
            x (Tensor): 输入到原始模型的张量。

        Returns:
            out (OrderedDict[str, Tensor]): 一个有序字典，包含了指定中间层的输出。
                                            key 是 `return_layers` 中指定的新名称。
        """
        out = OrderedDict()
        # 依次通过我们保存的层进行前向传播
        for name, module in self.items():
            x = module(x)
            # 如果当前层是我们想要捕获的层
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    torch.cat 的一个高效版本。如果输入列表中只有一个张量，它会直接返回该张量，
    避免了不必要的拷贝操作。

    Args:
        tensors (List[Tensor]): 需要拼接的张量列表。
        dim (int, optional): 拼接的维度。默认为 0。

    Returns:
        Tensor: 拼接后的张量。
    """
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    """
    一个检查函数，用于验证输入的 RoI (Region of Interest) boxes 的形状是否符合预期。
    RoI boxes 可以是两种格式之一：
    1. List[Tensor[L, 4]]: 一个列表，每个元素是一个张量，代表单张图片上的所有 proposal boxes。
    2. Tensor[K, 5]: 一个张量，其中 K 是整个批次的总 boxes 数量，格式为 [batch_index, x1, y1, x2, y2]。

    Args:
        boxes (Union[Tensor, List[Tensor]]): 需要检查的 RoI boxes。
    """
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            torch._assert(
                _tensor.size(1) == 4, "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
            )
    elif isinstance(boxes, torch.Tensor):
        torch._assert(boxes.size(1) == 5, "The boxes tensor shape is not correct as Tensor[K, 5]")
    else:
        torch._assert(False, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]")
    return


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    """
    将 proposals (proposals per image) 的列表转换为 RoI Align/Pool 层所期望的格式。
    具体来说，它将 `List[Tensor[N, 4]]` 转换为 `Tensor[K, 5]`，
    其中 K 是批次中所有框的总和，新增加的第一列是该框所属的图像在批次中的索引。

    Args:
        boxes (List[Tensor]): 一个列表，长度为批次大小。每个元素是一个形状为 [num_boxes, 4] 的张量，
                              代表一张图片上的所有边界框。

    Returns:
        Tensor: 一个形状为 [total_num_boxes, 5] 的张量，格式为 [batch_index, x1, y1, x2, y2]。
    """
    concat_boxes = _cat([b for b in boxes], dim=0)
    temp = [torch.full_like(b[:, :1], i) for i, b in enumerate(boxes)]
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def expand_boxes(boxes: Tensor, scale: float) -> Tensor:
    """
    将边界框从中心点向外扩展指定的比例。

    Args:
        boxes (Tensor): 形状为 `[N, 4]` 的边界框张量。
        scale (float): 扩展比例。

    Returns:
        Tensor: 扩展后的边界框张量。
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


@torch.jit.unused
def expand_masks_tracing_scale(M: int, padding: int) -> float:
    """
    在 ONNX 推理时，用于计算掩码扩展的尺度因子。
    它基于掩码的原始尺寸和填充量来计算。

    Args:
        M (int): 掩码的原始高度或宽度。
        padding (int): 四周要添加的填充量。

    Returns:
        float: 计算得到的尺度因子。
    """
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


def expand_masks(mask: Tensor, padding: int) -> Tuple[Tensor, float]:
    """
    在掩码的四周添加指定的 `padding`，并计算相应的尺度因子。
    这在 `paste_masks_in_image` 中用于在粘贴之前稍微放大掩码，以获得更平滑的边界。

    Args:
        mask (Tensor): 输入的掩码张量。
        padding (int): 四周要添加的填充量。

    Returns:
        Tuple[Tensor, float]:
            - padded_mask (Tensor): 填充后的掩码。
            - scale (float): 尺寸变化的比例。
    """
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # 兼容 ONNX tracing
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask: Tensor, box: Tensor, im_h: int, im_w: int) -> Tensor:
    """
    将一个较小的掩码（通常是模型预测的实例分割结果）粘贴到一个与原始图像大小相同的画布上。
    这个操作是 Mask R-CNN 后处理的关键步骤。

    Args:
        mask (Tensor): 要粘贴的掩码，通常尺寸较小 (例如 28x28)。
        box (Tensor): 掩码在目标图像上的位置，格式为 `[x1, y1, x2, y2]`。
        im_h (int): 目标图像的高度。
        im_w (int): 目标图像的宽度。

    Returns:
        Tensor: 一个与目标图像尺寸相同的张量，其中包含了被粘贴的掩码。
    """
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # 将掩码调整为目标边界框的大小
    mask = F.interpolate(mask[None, None], size=(h, w), mode="bilinear", align_corners=False)[0][0]

    # 创建一个空白的目标图像画布
    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    # 计算粘贴区域，并确保不越界
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    # 从调整大小后的掩码中相应区域裁剪并粘贴到画布上
    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    return im_mask


@torch.jit.unused
def _onnx_paste_mask_in_image(mask: Tensor, box: Tensor, im_h: Tensor, im_w: Tensor) -> Tensor:
    """ONNX-compatible version of paste_mask_in_image"""
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = box[2] - box[0] + one
    h = box[3] - box[1] + one
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]

    # TODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0, unpaded_im_mask.to(dtype=torch.float32), zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0, concat_0, zeros_x1), 1)[:, :im_w]
    return im_mask


@torch.jit.script
def _onnx_paste_masks_in_image_loop(masks: Tensor, boxes: Tensor, im_h: int, im_w: int) -> Tensor:
    """ONNX-compatible version of paste_masks_in_image_loop"""
    res_append = torch.zeros(0, im_h, im_w, dtype=masks.dtype, device=masks.device)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], torch.tensor(im_h), torch.tensor(im_w))
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append


def paste_masks_in_image(masks: Tensor, boxes: Tensor, img_shape: Tuple[int, int], padding: int = 1) -> Tensor:
    """
    将一批掩码粘贴到一批对应的边界框位置。

    Args:
        masks (Tensor): 形状为 `[N, 1, H, W]` 的掩码批次。
        boxes (Tensor): 形状为 `[N, 4]` 的边界框批次。
        img_shape (Tuple[int, int]): 目标图像的 `(H, W)`。
        padding (int): 传递给 `expand_masks` 的填充量。

    Returns:
        Tensor: 包含了所有粘贴后掩码的张量。
    """
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w)[:, None]

    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        return torch.stack(res, dim=0)[:, None]
    else:
        return masks.new_empty((0, 1, im_h, im_w))

# -------------------------------------------------------------------------------------------------
# | 训练辅助工具 (Training Helpers)                                                                |
# -------------------------------------------------------------------------------------------------

class BalancedPositiveNegativeSampler:
    """
    平衡正负样本采样器。
    在目标检测训练中，负样本（背景）的数量通常远超正样本（前景）。
    该类用于从大量的候选（如anchors）中采样一个固定大小、固定正负比例的小批次，
    以防止训练被大量的负样本主导，保证训练的稳定性和效率。
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        """
        Args:
            batch_size_per_image (int): 每张图片要采样的总样本数。
            positive_fraction (float): 采样的样本中，正样本所占的比例。
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        对匹配结果进行采样。

        Args:
            matched_idxs (List[Tensor]): Matcher的输出列表，每个张量对应一张图片。
                张量中的值: >= 1 表示正样本, 0 表示负样本, -1 表示忽略。

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                - pos_idx (List[Tensor]): 采样出的正样本的二进制掩码列表。
                - neg_idx (List[Tensor]): 采样出的负样本的二进制掩码列表。
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            # 找到所有正样本和负样本的索引
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            # 计算期望的正负样本数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # 保护：实际采样的正样本数不能超过存在的正样本数
            num_pos = min(positive.numel(), num_pos)
            
            num_neg = self.batch_size_per_image - num_pos
            # 保护：实际采样的负样本数不能超过存在的负样本数
            num_neg = min(negative.numel(), num_neg)

            # 随机打乱并采样指定数量的正负样本
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # 将采样到的索引转换为二进制掩码
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    将 proposal boxes 编码为相对于 reference_boxes (通常是 anchors) 的回归目标 (deltas)。
    这是边界框回归训练目标的核心计算。
    编码公式:
        tx = (x_gt - x_anchor) / w_anchor
        ty = (y_gt - y_anchor) / h_anchor
        tw = log(w_gt / w_anchor)
        th = log(h_gt / h_anchor)

    Args:
        reference_boxes (Tensor): 参考框 (如 anchors)。
        proposals (Tensor): 目标框 (如 ground-truth boxes)。
        weights (Tensor[4]): 用于加权 `(tx, ty, tw, th)` 的权重。

    Returns:
        Tensor: 编码后的回归目标 `(tx, ty, tw, th)`。
    """
    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder:
    """
    边界框编解码器。
    它封装了将边界框在 `(x1, y1, x2, y2)` 格式和回归器训练时使用的
    `(dx, dy, dw, dh)` 相对表示之间进行相互转换的逻辑。
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        """
        Args:
            weights (Tuple[float, float, float, float]): 用于编码和解码时加权的元组 (wx, wy, ww, wh)。
            bbox_xform_clip (float): 在解码时，对 `exp(dw)` 和 `exp(dh)` 的值进行裁剪，
                                     防止因预测值过大导致指数爆炸，从而产生无效的超大边界框。
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        """
        将一批 proposals 编码为相对于 reference_boxes 的表示。
        处理列表输入，适用于不同图片有不同数量框的场景。
        """
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return list(targets.split(boxes_per_image, 0))

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        对单个张量的 boxes 进行编码。
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        """
        将一批回归预测 (rel_codes) 解码为绝对坐标边界框。
        处理列表输入，适用于不同图片有不同数量框的场景。
        """
        torch._assert(
            isinstance(boxes, (list, tuple)),
            "This function expects boxes of type list or tuple.",
        )
        torch._assert(
            isinstance(rel_codes, torch.Tensor),
            "This function expects rel_codes of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = sum(val for val in boxes_per_image)
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        从原始 boxes (如 anchors) 和编码后的回归量 (rel_codes)，解码出预测的 boxes。
        这是 `encode_single` 的逆操作。

        Args:
            rel_codes (Tensor): 编码后的回归量 (dx, dy, dw, dh)。
            boxes (Tensor): 参考框 (如 anchors)。
        """
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        pred_boxes1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes2 = pred_ctr_y - 0.5 * pred_h
        pred_boxes3 = pred_ctr_x + 0.5 * pred_w
        pred_boxes4 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


class BoxLinearCoder:
    """
    The linear box-to-box transform defined in FCOS. The transformation is parameterized
    by the distance from the center of (square) src box to 4 edges of the target box.
    """

    def __init__(self, normalize_by_size: bool = True) -> None:
        """
        Args:
            normalize_by_size (bool): normalize deltas by the size of src (anchor) boxes.
        """
        self.normalize_by_size = normalize_by_size

    def encode(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded

        Returns:
            Tensor: the encoded relative box offsets that can be used to
            decode the boxes.

        """

        # get the center of reference_boxes
        reference_boxes_ctr_x = 0.5 * (reference_boxes[..., 0] + reference_boxes[..., 2])
        reference_boxes_ctr_y = 0.5 * (reference_boxes[..., 1] + reference_boxes[..., 3])

        # get box regression transformation deltas
        target_l = reference_boxes_ctr_x - proposals[..., 0]
        target_t = reference_boxes_ctr_y - proposals[..., 1]
        target_r = proposals[..., 2] - reference_boxes_ctr_x
        target_b = proposals[..., 3] - reference_boxes_ctr_y

        targets = torch.stack((target_l, target_t, target_r, target_b), dim=-1)

        if self.normalize_by_size:
            reference_boxes_w = reference_boxes[..., 2] - reference_boxes[..., 0]
            reference_boxes_h = reference_boxes[..., 3] - reference_boxes[..., 1]
            reference_boxes_size = torch.stack(
                (reference_boxes_w, reference_boxes_h, reference_boxes_w, reference_boxes_h), dim=-1
            )
            targets = targets / reference_boxes_size
        return targets

    def decode(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:

        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.

        Returns:
            Tensor: the predicted boxes with the encoded relative box offsets.

        .. note::
            This method assumes that ``rel_codes`` and ``boxes`` have same size for 0th dimension. i.e. ``len(rel_codes) == len(boxes)``.

        """

        boxes = boxes.to(dtype=rel_codes.dtype)

        ctr_x = 0.5 * (boxes[..., 0] + boxes[..., 2])
        ctr_y = 0.5 * (boxes[..., 1] + boxes[..., 3])

        if self.normalize_by_size:
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]

            list_box_size = torch.stack((boxes_w, boxes_h, boxes_w, boxes_h), dim=-1)
            rel_codes = rel_codes * list_box_size

        pred_boxes1 = ctr_x - rel_codes[..., 0]
        pred_boxes2 = ctr_y - rel_codes[..., 1]
        pred_boxes3 = ctr_x + rel_codes[..., 2]
        pred_boxes4 = ctr_y + rel_codes[..., 3]

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=-1)
        return pred_boxes


class Matcher:
    """
    匹配器。
    该类为每个预测元素（如 anchor box）分配一个真实元素（ground-truth box）。
    每个预测元素最多匹配一个真实元素；每个真实元素可以匹配零个或多个预测元素。

    匹配基于一个 MxN 的 `match_quality_matrix`（通常是 IoU 矩阵），
    该矩阵描述了每对（真实元素, 预测元素）的匹配质量。

    返回一个长度为 N 的张量，其中包含了每个预测框匹配到的真实框的索引。
    如果没有匹配，则返回一个负值。
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Args:
            high_threshold (float): 匹配质量 >= 该阈值的被视为正样本。
            low_threshold (float): 低于该阈值的被视为负样本。
            allow_low_quality_matches (bool): 是否允许“低质量匹配”。
                如果为 True，它会确保每个 ground-truth 至少被匹配到一个 anchor，
                即使该 anchor 与它的 IoU 小于 `high_threshold`。这对于召回难以检测的物体很重要。
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        根据 IoU 矩阵进行匹配。

        Args:
            match_quality_matrix (Tensor[float]): MxN 的张量，M是gt数量，N是anchor数量。

        Returns:
            matches (Tensor[int64]): 长度为 N 的张量。
                matches[i] 的值:
                - `k (>= 0)`: 第 `i` 个 anchor 匹配到了第 `k` 个 gt。
                - `-1 (BELOW_LOW_THRESHOLD)`: 负样本 (背景)。
                - `-2 (BETWEEN_THRESHOLDS)`: 忽略的样本。
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # 沿维度0（gt维度）取最大值，为每个 prediction 找到最佳的 gt 匹配
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # 根据阈值将匹配分为三类
        # 1. 高于 high_threshold (正样本) - 默认已标记
        # 2. 低于 low_threshold (负样本)
        below_low_threshold = matched_vals < self.low_threshold
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD

        # 3. 介于 low_threshold 和 high_threshold 之间 (忽略)
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            if all_matches is None:
                torch._assert(False, "all_matches should not be None")
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        """
        为只有低质量匹配的 gts 产生额外的匹配（如果 allow_low_quality_matches=True）。
        具体来说，对于每个 ground-truth，找到与之具有最大重叠的 prediction（可以有多个并列的）。
        对于这个集合中的每个 prediction，如果它当前是未匹配状态，则将其匹配到这个 ground-truth。
        """
        # 为每个 gt 找到其对应的最高匹配质量
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        
        # 找到所有拥有最高匹配质量的 (gt, prediction) 对
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


