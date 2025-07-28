from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch.nn.functional as F
from torch import nn, Tensor

from model.utils.misc_nn_ops import Conv2dNormActivation



class ExtraFPNBlock(nn.Module):
    """
    一个抽象基类，用于定义在 FPN 主体结构之上添加的额外层。
    FPN 的一个常见变体是在最高层特征图之后再接一个或多个下采样层，以产生更粗糙但语义更强的特征。
    例如，RetinaNet 和 EfficientDet 中使用的 `LastLevelMaxPool` 就是一个例子。
    继承这个类的模块需要实现 `forward` 方法。
    """
    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        """
        定义额外块的前向传播逻辑。

        Args:
            results (List[Tensor]): 由 FPN 主体生成的特征图列表。
            x (List[Tensor]): 从骨干网络输入的原始特征图列表。
            names (List[str]): FPN 生成的特征图的名称列表。

        Returns:
            Tuple[List[Tensor], List[str]]: 经过额外块处理后更新的特征图列表和名称列表。
        """
        pass


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络 (Feature Pyramid Network) 模块。
    FPN 通过自顶向下 (top-down) 的路径和横向连接 (lateral connections) 来增强
    标准卷积网络（如ResNet）的特征层次结构。它将高层级的强语义特征与低层级的
    高分辨率特征相结合，为不同尺度的目标检测和分割生成丰富的多尺度特征图。

    该模块的输入是来自骨干网络不同阶段的特征图（一个有序字典），输出也是一个有序字典，
    包含了经过 FPN 处理后的、具有相同通道数的多尺度特征图。

    Args:
        in_channels_list (List[int]): 一个列表，包含了来自骨干网络每个阶段的输入特征图的通道数。
                                      例如，ResNet-50 的 [C2, C3, C4, C5] 对应的通道数可能是 [256, 512, 1024, 2048]。
        out_channels (int): FPN 输出的所有特征图的统一通道数。这是一个超参数，通常设为 256。
        extra_blocks (Optional[ExtraFPNBlock]): 一个可选的模块，用于在FPN的最高层特征之上添加额外的层。
                                                 默认为 None。
        norm_layer (Optional[Callable[..., nn.Module]]): 在 FPN 的卷积层后使用的归一化层。默认为 None。
    """

    _version = 2

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        
        # `inner_blocks` 用于处理横向连接 (lateral connection)。
        # 它们是 1x1 卷积，用于将骨干网络输出的特征图通道数统一到 `out_channels`。
        self.inner_blocks = nn.ModuleList()
        # `layer_blocks` 用于在横向连接和自顶向下路径融合后进行处理。
        # 它们是 3x3 卷积，用于平滑融合后的特征，减少上采样带来的混叠效应。
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            
            # 1x1 卷积，用于横向连接
            inner_block_module = Conv2dNormActivation(
                in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, activation_layer=None
            )
            # 3x3 卷积，用于平滑输出
            layer_block_module = Conv2dNormActivation(
                out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, activation_layer=None
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # 初始化 FPN 模块的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 检查 extra_blocks 的类型是否正确
        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}")
        self.extra_blocks = extra_blocks

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
        # 为了兼容旧版本的 state_dict，进行键名转换
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            num_blocks = len(self.inner_blocks)
            for block in ["inner_blocks", "layer_blocks"]:
                for i in range(num_blocks):
                    for type in ["weight", "bias"]:
                        old_key = f"{prefix}{block}.{i}.{type}"
                        new_key = f"{prefix}{block}.{i}.0.{type}" # ConvNormActivation 是一个 Sequential，层在索引0
                        if old_key in state_dict:
                            state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        等价于 self.inner_blocks[idx](x)，但 TorchScript 不支持直接索引 ModuleList。
        因此使用循环来实现。
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        等价于 self.layer_blocks[idx](x)，原因同上。
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        执行 FPN 的前向传播。

        Args:
            x (Dict[str, Tensor]): 一个有序字典，包含了从骨干网络不同层级输出的特征图。
                                   键是层的名称，值是对应的特征图张量。
                                   例如：{'feat0': c2, 'feat1': c3, 'feat2': c4, 'feat3': c5}

        Returns:
            results (Dict[str, Tensor]): 一个有序字典，包含了 FPN 生成的多尺度特征图。
                                        键是特征图的名称，值是对应的张量。
                                        顺序从高分辨率到低分辨率。
        """
        # 将输入的字典解包成名称列表和张量列表，方便处理
        names = list(x.keys())
        x = list(x.values())

        # 1. 自顶向下路径 (Top-down pathway)
        # 首先处理最高层（分辨率最低，语义最强）的特征图
        # 通过 1x1 卷积（inner_block）进行横向连接
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        
        results = []
        # 通过 3x3 卷积（layer_block）处理后，作为 FPN 的最高层输出
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        # 从次高层开始，逆向遍历骨干网络的特征图
        for idx in range(len(x) - 2, -1, -1):
            # 获取当前层的特征图，并通过 1x1 卷积进行横向连接
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            
            # 将上一层（更高层级）的特征图进行上采样，使其与当前层分辨率匹配
            # 使用最近邻插值，这是 FPN 论文中提到的方法
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            
            # 元素级相加，融合横向连接的特征和自顶向下的特征
            last_inner = inner_lateral + inner_top_down
            
            # 通过 3x3 卷积处理融合后的特征，并将其插入到结果列表的开头（以保持从高分辨率到低分辨率的顺序）
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # 2. （可选）添加额外的 FPN 块
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # 3. 将结果重新打包成一个有序字典
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    一个实现了 `ExtraFPNBlock` 的模块，它在 FPN 的最高层特征图之上应用了一个最大池化层。
    这是一种简单而有效的方法，用于从最高层特征（例如P5）生成一个分辨率更低的特征图（例如P6）。
    这在一些目标检测模型（如 RetinaNet）中被用来覆盖更大范围的锚点（anchors）。
    """
    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        """
        在前向传播中，将池化后的结果添加到特征图列表和名称列表中。

        Args:
            x (List[Tensor]): FPN 主体生成的特征图列表 (P2, P3, P4, P5)。
            y (List[Tensor]): 骨干网络输入的原始特征图列表 (C2, C3, C4, C5)。此参数在此处未使用。
            names (List[str]): FPN 特征图的名称列表。

        Returns:
            Tuple[List[Tensor], List[str]]: 更新后的特征图列表和名称列表。
        """
        # 为新的特征图添加名称 "pool"
        names.append("pool")
        # 对 FPN 的最后一个输出（最高层特征图）应用最大池化
        # kernel_size=1, stride=2 意味着将特征图的尺寸减半
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names
