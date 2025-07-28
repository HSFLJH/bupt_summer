# 该文件定义了在将数据送入通用 R-CNN 模型（如 Faster R-CNN, Mask R-CNN）前后
# 所需执行的一系列转换操作。这些操作被封装在 `GeneralizedRCNNTransform` 类中，
# 它是模型不可或缺的一部分，负责处理图像和标注的标准化、尺寸调整和批处理。
#
# 主要功能包括：
# 1. 图像标准化: 减去均值，除以标准差。
# 2. 尺寸调整 (Resize): 将输入图像和其标注（边界框、掩码、关键点）调整到
#    模型期望的尺寸范围 (由 `min_size` 和 `max_size` 控制)。
# 3. 图像批处理 (Batching): 将多张可能尺寸不同的图像打包成一个统一尺寸的批次张量，
#    通过在右侧和下方进行填充 (padding) 实现。
# 4. 后处理 (Post-processing): 将模型的预测结果（边界框、掩码等）从模型推理时的
#    图像尺寸反向映射回原始图像尺寸。

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
import torchvision
from model.utils.anchor import ImageList
from model.utils.general import paste_masks_in_image


def _get_shape_onnx(image: Tensor) -> Tensor:
    """ONNX-compatible version of getting image shape."""
    from torch.onnx import operators

    return operators.shape_as_tensor(image)[-2:]


def _fake_cast_onnx(v: Tensor) -> float:
    # ONNX requires a tensor but here we fake its type for JIT.
    return v


def _resize_image_and_masks(
    image: Tensor,
    self_min_size: float,
    self_max_size: float,
    target: Optional[Dict[str, Tensor]] = None,
    fixed_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    """内部函数，用于调整单个图像和其对应掩码的尺寸。"""
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        if torchvision._is_tracing():
            scale_factor = _fake_cast_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True

    image = torch.nn.functional.interpolate(
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    if target is None:
        return image, target

    # 相应地调整掩码的尺寸
    if "masks" in target:
        mask = target["masks"]
        mask = torch.nn.functional.interpolate(
            mask[:, None].float(), size=size, scale_factor=scale_factor, recompute_scale_factor=recompute_scale_factor
        )[:, 0].byte()
        target["masks"] = mask
    return image, target


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    """
    根据图像尺寸的变化，相应地调整边界框的坐标。

    Args:
        boxes (Tensor): `[N, 4]` 的边界框张量。
        original_size (List[int]): 原始图像尺寸 `[H, W]`。
        new_size (List[int]): 新的图像尺寸 `[H_new, W_new]`。

    Returns:
        Tensor: 调整尺寸后的边界框张量。
    """
    ratios = [
        torch.tensor(s_new, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s_new, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def resize_keypoints(keypoints: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    """
    根据图像尺寸的变化，相应地调整关键点的坐标。

    Args:
        keypoints (Tensor): `[N, K, 3]` 的关键点张量，每行格式为 `(x, y, visibility)`。
        original_size (List[int]): 原始图像尺寸 `[H, W]`。
        new_size (List[int]): 新的图像尺寸 `[H_new, W_new]`。

    Returns:
        Tensor: 调整尺寸后的关键点张量。
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=keypoints.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data


class GeneralizedRCNNTransform(nn.Module):
    """
    在将数据送入 GeneralizedRCNN 模型之前，执行输入/目标的转换。

    执行的转换包括:
        - 输入标准化 (减均值和除以标准差)
        - 输入/目标尺寸调整，以匹配 `min_size` / `max_size`
        - 将图像批处理为 ImageList 对象

    它返回一个用于输入的 ImageList 和一个用于目标的 List[Dict[Tensor]]。
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        image_mean: List[float],
        image_std: List[float],
        size_divisible: int = 32,
        fixed_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            min_size (int): 调整后图像的最小边长。
            max_size (int): 调整后图像的最大边长。
            image_mean (List[float]): 用于标准化的图像均值。
            image_std (List[float]): 用于标准化的图像标准差。
            size_divisible (int): 确保批处理后的图像尺寸可以被该值整除。这对于FPN等网络是必要的。
            fixed_size (Optional[Tuple[int, int]]): 如果提供，则将图像直接调整到此固定尺寸。
        """
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size
        self._skip_resize = kwargs.pop("_skip_resize", False)

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        """
        对一批图像和它们对应的标注进行预处理。

        Args:
            images (List[Tensor]): `[C, H, W]` 格式的图像列表。
            targets (Optional[List[Dict[str, Tensor]]]): 标注列表，每个字典包含 "boxes", "labels" 等。

        Returns:
            Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
                - image_list: 包含批处理后图像和其尺寸的 `ImageList` 对象。
                - targets: 经过同样尺寸调整的标注列表。
        """
        images = [img for img in images]
        if targets is not None:
            # 创建 targets 的副本以避免原地修改
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {k: v for k, v in t.items()}
                targets_copy.append(data)
            targets = targets_copy

        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            
            # 1. 标准化图像
            image = self.normalize(image)
            # 2. 调整图像和标注的尺寸
            image, target_index = self.resize(image, target_index)
            
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录下每张图像调整后的尺寸
        image_sizes = [img.shape[-2:] for img in images]
        # 3. 将图像批处理成一个大张量
        images = self.batch_images(images, size_divisible=self.size_divisible)
        
        image_sizes_list: List[Tuple[int, int]] = [
            (s[0], s[1]) for s in image_sizes
        ]

        # 封装成 ImageList 对象
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image: Tensor) -> Tensor:
        """
        通过减去均值并除以标准差来标准化图像。
        """
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k: List[int]) -> int:
        """
        通过 torch 操作实现 `random.choice`，使其可被 TorchScript 编译。
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
        return k[index]

    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        调整单张图像及其标注的尺寸。
        在训练时，会从 `self.min_size` 中随机选择一个尺寸。
        在测试时，通常使用 `self.min_size` 中的最大值。
        图像的另一边会按比例缩放，但不能超过 `self.max_size`。
        """
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            # 训练时，随机选择一个 min_size
            size = float(self.torch_choice(self.min_size))
        else:
            # 测试时，使用最大的 min_size
            size = float(self.min_size[-1])
        
        # 调整图像和掩码
        image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target

        # 调整边界框
        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        # 调整关键点
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        """
        将一个图像列表打包成一个批次。
        它会找到这批图像中的最大高度和最大宽度，然后将所有图像都填充到这个尺寸。
        填充后的尺寸还会被调整，以确保可以被 `size_divisible` 整除。

        Args:
            images (List[Tensor]): 要批处理的图像列表。
            size_divisible (int): 最终尺寸需要能被此数整除。

        Returns:
            Tensor: 批处理后的张量。
        """
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        # 找到批次中的最大 H 和 W
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        # 向上取整，确保 H 和 W 是 stride 的倍数
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # 创建一个空白的批次张量
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        # 将每张图复制到批次张量的对应位置
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        """
        对模型的输出进行后处理，将预测结果从模型推理时的尺寸映射回原始图像尺寸。

        Args:
            result (List[Dict[str, Tensor]]): 模型的预测输出列表。
            image_shapes (List[Tuple[int, int]]): 批处理中，每张图像在输入模型前的尺寸。
            original_image_sizes (List[Tuple[int, int]]): 每张图像的原始尺寸。

        Returns:
            List[Dict[str, Tensor]]: 后处理过的预测结果。
        """
        if self.training:
            return result
        
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        _indent = "\n    "
        format_string += f"{_indent}Normalize(mean={self.image_mean}, std={self.image_std})"
        format_string += f"{_indent}Resize(min_size={self.min_size}, max_size={self.max_size}, mode='bilinear')"
        format_string += "\n)"
        return format_string
