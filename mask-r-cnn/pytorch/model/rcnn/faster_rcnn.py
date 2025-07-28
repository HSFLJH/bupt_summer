# 该文件定义了 Faster R-CNN 模型。
#
# Faster R-CNN 是一个经典的两阶段目标检测器。该文件通过继承 `GeneralizedRCNN` 类，
# 将其各个组件（如 RPN、RoI Heads）进行具体的实例化和组装，从而构建出完整的
# Faster R-CNN 模型。
#
# 这个实现是高度模块化和可配置的，允许用户轻松地替换或调整模型的各个部分，
# 例如使用不同的骨干网络、锚框生成器或 RoI 头部。

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from model.utils.poolers import MultiScaleRoIAlign
from model.utils.anchor import AnchorGenerator
from model.rcnn.general_rcnn import GeneralizedRCNN
from model.rpn.rpn import RPNHead, RegionProposalNetwork
from model.utils.transforms import GeneralizedRCNNTransform
from model.heads.roi_head import RoIHeads
from torch import Tensor

def _default_anchorgen() -> AnchorGenerator:
    """
    为 FPN 创建一个默认的锚框生成器 (AnchorGenerator)。
    它为 FPN 的5个不同尺度的特征图分别定义了锚框的尺寸。
    """
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class FasterRCNN(GeneralizedRCNN):
    """
    实现了 Faster R-CNN 模型。

    这个模型结合了特征提取骨干网络、RPN 和 RoI Heads，用于目标检测。
    它继承自 `GeneralizedRCNN`，并通过具体的模块配置来定义其行为。

    Args:
        backbone: 特征提取网络。
        num_classes (optional): 类别数（包括背景）。
        min_size, max_size (int): 图像预处理时的最小和最大尺寸。
        image_mean, image_std (list[float]): 图像标准化的均值和标准差。
        rpn_... : RPN 相关的配置参数。
        box_... : RoI Heads (Box Head) 相关的配置参数。
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
        rpn_anchor_generator: Optional[AnchorGenerator] = None,
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
        box_roi_pool: Optional[MultiScaleRoIAlign] = None,
        box_head: Optional[nn.Module] = None,
        box_predictor: Optional[nn.Module] = None,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        bbox_reg_weights: Optional[Tuple[float, float, float, float]] = None,
        **kwargs: Any,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        # --------- 1. RPN 组件的构建 ---------
        out_channels = backbone.out_channels

        # 如果未提供锚框生成器，则使用默认的
        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        
        # 如果未提供 RPN 头部，则创建一个标准的
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        # 为训练和测试模式分别定义 NMS 前后保留的 proposal 数量
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 实例化 RPN
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        # --------- 2. RoI Heads 组件的构建 ---------
        # 如果未提供 RoI 池化层，则创建一个默认的 MultiScaleRoIAlign
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        # 如果未提供 Box Head，则创建一个默认的两层 MLP
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        # 如果未提供 Box Predictor，则创建一个默认的 FastRCNNPredictor
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        # 实例化 RoIHeads
        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        # --------- 3. Transform 组件的构建 ---------
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        # 调用父类 `GeneralizedRCNN` 的构造函数，传入组装好的四大组件
        super().__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    用于 FPN 模型中 Fast R-CNN部分的标准检测头。
    它接收经过 RoI Pooling 后的特征，并通过两个全连接层（MLP）
    来提取用于最终分类和回归的高维表示。

    Args:
        in_channels (int): 输入特征的通道数。
        representation_size (int): 中间表示层的大小。
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    用于 Fast R-CNN 的标准分类 + 边界框回归层。
    它接收 `TwoMLPHead` 提取的特征，并使用两个并行的线性层来分别
    输出最终的类别得分和边界框回归量。

    Args:
        in_channels (int): 输入特征的通道数。
        num_classes (int): 输出的类别数（包括背景类）。
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
