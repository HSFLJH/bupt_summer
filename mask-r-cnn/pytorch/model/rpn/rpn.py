# 该文件定义了区域提议网络 (Region Proposal Network, RPN) 的相关组件。
# RPN 是两阶段目标检测器（如 Faster R-CNN, Mask R-CNN）的核心，
# 它的主要职责是接收来自骨干网络（如 FPN）的特征图，并生成一系列可能包含物体的候选区域（proposals）。

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from model.utils.misc_nn_ops import Conv2dNormActivation
import model.utils.boxes as box_utils
from model.utils.anchor import AnchorGenerator
from model.utils.anchor import ImageList
import model.utils.general as general_utils


# -------------------------------------------------------------------------------------------------
# | RPN 头部 (Head)                                                                                |
# -------------------------------------------------------------------------------------------------

class RPNHead(nn.Module):
    """
    一个简单的 RPN 头部模块。
    它接收 FPN 输出的每个层级的特征图，并通过一个卷积层和两个并行的卷积层，
    分别为每个锚框预测其“物体性”得分（即是前景还是背景）和边界框回归量。

    Args:
        in_channels (int): 输入特征图的通道数。
        num_anchors (int): 每个空间位置上的锚框数量。
        conv_depth (int, optional): 在分类和回归头之前的共享卷积层数量。默认为 1。
    """
    _version = 2

    def __init__(self, in_channels: int, num_anchors: int, conv_depth: int = 1) -> None:
        super().__init__()
        
        # 一个共享的卷积层，用于在预测前进一步处理特征
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        
        # 分类头：一个 1x1 卷积，输出通道数为 num_anchors，用于预测每个锚框的物体性得分
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        
        # 回归头：一个 1x1 卷积，输出通道数为 num_anchors * 4，用于预测每个锚框的4个回归参数 (dx, dy, dw, dh)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # 初始化权重
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

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
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
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


    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        RPNHead 的前向传播。

        Args:
            x (List[Tensor]): 一个包含了 FPN 各个层级特征图的列表。

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                - 一个包含了每个层级预测的物体性得分的列表。
                - 一个包含了每个层级预测的边界框回归量的列表。
        """
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


# -------------------------------------------------------------------------------------------------
# | 辅助函数 (Helper Functions)                                                                  |
# -------------------------------------------------------------------------------------------------

def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    """
    对 RPNHead 的输出张量进行维度重排和展平。
    原始形状: [N, A*C, H, W]  (批次大小, 锚框数*类别数, 高, 宽)
    目标形状: [N, H*W*A, C]  (批次大小, 总锚框数, 类别数)
    这样做是为了方便后续的计算（如损失计算和与 anchor 的匹配）。

    Args:
        layer (Tensor): 输入的张量。
        N (int): 批次大小。
        A (int): 每个位置的锚框数量。
        C (int): 每个锚框的预测类别数或回归参数数 (对于分类是1，对于回归是4)。
        H (int): 特征图高度。
        W (int): 特征图宽度。

    Returns:
        Tensor: 重排并展平后的张量。
    """
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, A, C]
    layer = layer.reshape(N, -1, C)      # [N, H*W*A, C]
    return layer


def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    将来自 FPN 所有层级的预测结果进行拼接。

    Args:
        box_cls (List[Tensor]): 每个 FPN 层级的分类预测列表。
        box_regression (List[Tensor]): 每个 FPN 层级的回归预测列表。

    Returns:
        Tuple[Tensor, Tensor]:
            - 拼接后的总分类预测张量，形状为 `[N*K, 1]`，K是所有层级的总锚框数。
            - 拼接后的总回归预测张量，形状为 `[N*K, 4]`。
    """
    box_cls_flattened = []
    box_regression_flattened = []
    
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        
        # 重排和展平分类预测
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # 重排和展平回归预测
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    
    # 在特征层级维度（dim=1）上进行拼接
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


# -------------------------------------------------------------------------------------------------
# | 区域提议网络 (Region Proposal Network)                                                          |
# -------------------------------------------------------------------------------------------------

class RegionProposalNetwork(torch.nn.Module):
    """
    实现了 RPN (区域提议网络)。

    Args:
        anchor_generator (AnchorGenerator): 用于生成锚框的模块。
        head (nn.Module): RPN 头部，用于进行分类和回归预测。
        fg_iou_thresh (float): IoU 大于此阈值的锚框被视为正样本（前景）。
        bg_iou_thresh (float): IoU 小于此阈值的锚框被视为负样本（背景）。
        batch_size_per_image (int): 每张图片用于训练 RPN 的锚框样本数量。
        positive_fraction (float): 在每个样本批次中，正样本所占的比例。
        pre_nms_top_n (Dict[str, int]): 在 NMS（非极大值抑制）之前，每个特征层级保留的最高分 proposals 的数量。
                                       字典的 key 是 'training' 和 'testing'。
        post_nms_top_n (Dict[str, int]): 在 NMS 之后，最终输出的 proposals 的数量。
                                        字典的 key 是 'training' 和 'testing'。
        nms_thresh (float): NMS 使用的 IoU 阈值。
        score_thresh (float, optional): 用于在 NMS 前过滤低分框的得分阈值。默认为 0.0。
    """
    __annotations__ = {
        "box_coder": general_utils.BoxCoder,
        "proposal_matcher": general_utils.Matcher,
        "fg_bg_sampler": general_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # RPN 训练参数
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # RPN 推理参数
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = general_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # --- 训练时使用的组件 ---
        self.box_similarity = box_utils.box_iou # 用于计算 IoU 的函数
        # 用于将锚框与真实边界框（ground-truth boxes）进行匹配
        self.proposal_matcher = general_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True, # 确保每个 gt_box 至少有一个 anchor 匹配
        )
        # 用于在正负样本中进行采样，以构建一个平衡的小批次来计算损失
        self.fg_bg_sampler = general_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        
        # --- 推理时使用的组件 ---
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        """根据是训练还是测试模式，返回 NMS 前保留的 top-N 数量。"""
        return self._pre_nms_top_n["training"] if self.training else self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        """根据是训练还是测试模式，返回 NMS 后保留的 top-N 数量。"""
        return self._post_nms_top_n["training"] if self.training else self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        为锚框分配训练目标（标签和回归目标）。【仅在训练时调用】

        Args:
            anchors (List[Tensor]): 每张图片的锚框列表。
            targets (List[Dict[str, Tensor]]): 每张图片的真实标注列表。

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                - labels (List[Tensor]): 每个锚框的标签 (1: 前景, 0: 背景, -1: 忽略)。
                - matched_gt_boxes (List[Tensor]): 每个锚框匹配到的真实边界框。
        """
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # 处理没有真实物体的图片
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算 gt_boxes 和 anchors 之间的 IoU 矩阵
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                # 根据 IoU 矩阵为每个 anchor 匹配一个 gt_box 的索引
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                
                # 获取每个 anchor 匹配到的 gt_box 坐标
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # 生成标签：IoU > fg_thresh 的为正样本 (1)
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # IoU < bg_thresh 的为负样本 (0)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # 介于两者之间的样本在计算损失时被忽略 (-1)
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        """从每个 FPN 层级独立地选出 top-N 个得分最高的锚框索引。"""
        r = []
        offset = 0
        # `split` 会将 objectness 张量按每个层级的锚框数量进行切分
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            # 确保 top-N 不超过该层级的总锚框数
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            # 获取 top-N 的索引
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        对解码后的 proposals 进行过滤、裁剪和 NMS 操作，以生成最终的候选区域。

        Args:
            proposals (Tensor): 解码后的候选框，形状为 `[N, K, 4]`。
            objectness (Tensor): 物体性得分，形状为 `[N, K]`。
            image_shapes (List[Tuple[int, int]]): 批次中每张图片的原始尺寸。
            num_anchors_per_level (List[int]): 每个 FPN 层级的锚框数量。

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                - final_boxes (List[Tensor]): 每张图片最终的候选框列表。
                - final_scores (List[Tensor]): 每个候选框对应的得分列表。
        """
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach() # 不对 objectness 进行反向传播
        objectness = objectness.reshape(num_images, -1)

        # 为每个锚框分配一个层级索引
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0).reshape(1, -1).expand_as(objectness)

        # 1. 在 NMS 之前，按层级独立选出 top-N
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        
        # 提取 top-N 的 proposals, scores, 和 levels
        proposals = proposals[batch_idx, top_n_idx]
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        
        # 将 logits 转换为概率
        objectness_prob = torch.sigmoid(objectness)

        final_boxes, final_scores = [], []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 2. 将 proposals 裁剪到图像边界内
            boxes = box_utils.clip_boxes_to_image(boxes, img_shape)

            # 3. 移除尺寸过小的 proposals
            keep = box_utils.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 4. 移除得分过低的 proposals (可选)
            if self.score_thresh > 0.0:
                keep = torch.where(scores >= self.score_thresh)[0]
                boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 5. 按层级独立执行 NMS
            keep = box_utils.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # 6. 保留 NMS 后的 top-N proposals
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        计算 RPN 的损失。【仅在训练时调用】
        损失包括：物体性分类损失 (二元交叉熵) 和边界框回归损失 (Smooth L1)。

        Args:
            objectness (Tensor): 拼接后的物体性预测。
            pred_bbox_deltas (Tensor): 拼接后的边界框回归预测。
            labels (List[Tensor]): 分配给每个锚框的标签列表。
            regression_targets (List[Tensor]): 为每个锚框编码的回归目标列表。

        Returns:
            Tuple[Tensor, Tensor]:
                - objectness_loss: 物体性损失。
                - box_loss: 边界框回归损失。
        """
        # 采样正负样本以计算损失
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        
        # 计算回归损失 (只对正样本计算)
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel()) # 按总样本数归一化

        # 计算分类损失 (对所有采样样本计算)
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        RPN 的主前向传播函数。

        Args:
            images (ImageList): 输入的图像批次。
            features (Dict[str, Tensor]): 来自 FPN 的特征图字典。
            targets (Optional[List[Dict[str, Tensor]]]): 真实标注，仅在训练时提供。

        Returns:
            Tuple[List[Tensor], Dict[str, Tensor]]:
                - boxes (List[Tensor]): 每张图片生成的候选区域列表。
                - losses (Dict[str, Tensor]): 包含损失的字典，仅在训练时非空。
        """
        features = list(features.values())
        # 1. 通过 RPN Head 得到预测
        objectness, pred_bbox_deltas = self.head(features)
        
        # 2. 生成锚框
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level = [o.shape[-2] * o.shape[-1] * o.shape[-3] for o in objectness] # C*H*W, C=num_anchors
        
        # 3. 拼接所有层级的预测结果
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        
        # 4. 将预测的回归量应用到锚框上，解码出 proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        
        # 5. 过滤 proposals
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None during training")
            # 6. (训练时) 为锚框分配目标
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 7. (训练时) 为回归损失计算目标
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            # 8. (训练时) 计算损失
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses


