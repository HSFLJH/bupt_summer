# 该文件定义了 Mask R-CNN 模型。
#
# Mask R-CNN 在 Faster R-CNN 的基础上进行了扩展，增加了一个并行的分支，
# 用于对每个检测到的物体（RoI）进行像素级的实例分割，即生成一个二值掩码（mask）。
#
# 这个实现通过继承 `FasterRCNN` 类来重用其大部分的目标检测逻辑（骨干网络、RPN、
# 边界框回归和分类头），并在此之上添加了专用于掩码预测的组件：
# 1. `mask_roi_pool`: 一个新的 `MultiScaleRoIAlign` 层，通常具有比 box_roi_pool 更高的
#    分辨率（如 14x14），用于从特征图上为每个 RoI 提取更精细的特征。
# 2. `mask_head`: 一个由多个卷积层构成的全卷积网络 (FCN)，用于在提取出的特征上进行处理。
# 3. `mask_predictor`: 一个最终的卷积层（通常是转置卷积），将 `mask_head` 的输出
#    上采样并预测出每个类别的二值掩码。

from model.rcnn.faster_rcnn import FasterRCNN
from typing import Optional, Callable, List
from torch import nn, Tensor
import model.utils.misc_nn_ops as misc_nn_ops
from model.utils.poolers import MultiScaleRoIAlign
from collections import OrderedDict


class MaskRCNN(FasterRCNN):
    """
    实现了 Mask R-CNN 模型。

    这个模型在 Faster R-CNN 的基础上增加了一个掩码预测分支。
    它继承自 `FasterRCNN`，并在其 `roi_heads` 中添加了 `mask_roi_pool`,
    `mask_head`, 和 `mask_predictor`。

    Args:
        backbone: 特征提取网络。
        num_classes (optional): 类别数（包括背景）。
        (...其他 Faster R-CNN 参数)
        mask_roi_pool (optional): 用于掩码分支的 RoIAlign 层。
        mask_head (optional): 掩码预测头 (FCN)。
        mask_predictor (optional): 最终的掩码预测层。
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: Optional[int] = None,
        # transform parameters
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # RPN parameters
        rpn_anchor_generator: Optional[nn.Module] = None,
        rpn_head: Optional[nn.Module] = None,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_score_thresh: float = 0.0,
        # Box parameters
        box_roi_pool: Optional[nn.Module] = None,
        box_head: Optional[nn.Module] = None,
        box_predictor: Optional[nn.Module] = None,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        bbox_reg_weights: Optional[List[float]] = None,
        # Mask parameters
        mask_roi_pool: Optional[nn.Module] = None,
        mask_head: Optional[nn.Module] = None,
        mask_predictor: Optional[nn.Module] = None,
        **kwargs,
    ):

        if not isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(mask_roi_pool)}"
            )

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        # --------- 掩码分支组件的构建 ---------

        # 如果未提供掩码 RoI 池化层，则创建一个默认的
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        # 如果未提供掩码头，则创建一个默认的
        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        # 如果未提供掩码预测器，则创建一个默认的
        if mask_predictor is None:
            mask_predictor_in_channels = 256  # 通常等于 mask_head 的输出通道
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        # 调用父类 `FasterRCNN` 的构造函数来构建基础的目标检测部分
        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            **kwargs,
        )

        # 将构建好的掩码分支组件添加到 `roi_heads` 中
        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor


class MaskRCNNHeads(nn.Sequential):
    """
    标准的掩码头，由一系列的 FCN (全卷积网络) 层构成。
    它接收从 `mask_roi_pool` 提取的特征，并进行深度的非线性变换，
    为最终的掩码预测做准备。

    Args:
        in_channels (int): 输入特征的通道数。
        layers (List[int]): 每个 FCN 层的特征维度（输出通道数）。
        dilation (int): 卷积核的膨胀率。
        norm_layer (callable, optional): 使用的归一化层。默认为 None。
    """
    _version = 2

    def __init__(self, in_channels: int, layers: List[int], dilation: int, norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(
                misc_nn_ops.Conv2dNormActivation(
                    next_feature,
                    layer_features,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
            next_feature = layer_features

        super().__init__(*blocks)
        # 初始化权重
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            num_blocks = len(self)
            for i in range(num_blocks):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}mask_fcn{i+1}.{type}"
                    new_key = f"{prefix}{i}.0.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MaskRCNNPredictor(nn.Sequential):
    """
    标准的掩码预测器。
    它通常由一个转置卷积层和一个1x1卷积层组成。
    - 转置卷积: 将 `MaskRCNNHeads` 输出的特征图进行上采样（通常是2倍），
                使其分辨率变得更高，有利于生成更精细的掩码。
    - 1x1 卷积: 将特征图的通道数变为 `num_classes`，从而为每个类别预测一个掩码。

    Args:
        in_channels (int): 输入特征的通道数 (来自 MaskRCNNHeads)。
        dim_reduced (int): 转置卷积后的中间层通道数。
        num_classes (int): 最终输出的类别数。
    """
    def __init__(self, in_channels: int, dim_reduced: int, num_classes: int):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        # 初始化权重
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0)
