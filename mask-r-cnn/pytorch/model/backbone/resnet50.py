# 我们的骨干网络，主要是resnet50
from typing import Any, Callable, List, Optional, Type, Union, Dict, Tuple
from torch import nn, Tensor
import torch
from model.backbone.fpn import ExtraFPNBlock, LastLevelMaxPool, FeaturePyramidNetwork
from model.utils.general import IntermediateLayerGetter

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    创建一个 3x3 的卷积层，带有固定的 padding。
    这是 ResNet 中最常用的卷积核尺寸之一。

    Args:
        in_planes (int): 输入通道数。
        out_planes (int): 输出通道数。
        stride (int, optional): 卷积步长。默认为 1。
        groups (int, optional): 分组卷积的组数。默认为 1。
        dilation (int, optional): 卷积核的膨胀率。默认为 1。

    Returns:
        nn.Conv2d: 一个配置好的 3x3 卷积层。
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    创建一个 1x1 的卷积层。
    在 ResNet 的 Bottleneck 结构中，1x1 卷积常用于改变通道数（升维或降维），以及在下采样路径中匹配维度。

    Args:
        in_planes (int): 输入通道数。
        out_planes (int): 输出通道数。
        stride (int, optional): 卷积步长。默认为 1。

    Returns:
        nn.Conv2d: 一个配置好的 1x1 卷积层。
    """
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=stride, 
        bias=False
    )


class BasicBlock(nn.Module):
    """
    ResNet 的基础残差块 (Basic Residual Block)。
    主要用于较浅的 ResNet 网络，如 ResNet-18 和 ResNet-34。
    结构为：
        conv3x3 -> bn -> relu
        conv3x3 -> bn
        然后将输出与 identity（输入）相加，再通过 relu 激活。
    """
    expansion: int = 1  # BasicBlock 不会改变输出通道数，扩展因子为1。

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # 主路径的第一个卷积层，可能会进行下采样（当 stride != 1 时）
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # 主路径的第二个卷积层
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # 下采样层，用于当 identity 的维度与主路径输出不匹配时调整 identity
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 主路径前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要，对 identity 进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：将主路径的输出与 identity 相加
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet 的瓶颈残差块 (Bottleneck Residual Block)。
    主要用于更深的网络，如 ResNet-50, 101, 152，计算效率更高。
    结构为：
        conv1x1 -> bn -> relu  (降维)
        conv3x3 -> bn -> relu  (特征提取，可能会有 stride)
        conv1x1 -> bn          (升维)
    然后将输出与 identity 相加，再通过 relu 激活。
    """
    # Bottleneck 块的输出通道数是输入 planes 的 4 倍
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        # 主路径
        # 第一个 1x1 卷积，用于降维（从 inplanes 到 width）
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 3x3 卷积，用于特征提取，可能会进行下采样（当 stride != 1 时）
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 第二个 1x1 卷积，用于升维（从 width 到 planes * self.expansion）
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # 下采样层，用于 identity
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 主路径前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要，对 identity 进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    通用的 ResNet 模型实现。
    可以根据传入的 block 类型和 layers 列表来构建不同深度的 ResNet。
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # 第一个卷积层后的输出通道数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 元组中的每个元素指示是否应将该阶段的 2x2 stride 替换为膨胀卷积
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        
        # Stage 0: 初始的输入层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 1-4: 构建四个残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # 分类头 (Classification Head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 零初始化残差分支的最后一个BN层，这可以提高模型性能
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """
        构建 ResNet 的一个 "层" (stage)，由多个残差块堆叠而成。

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): 要使用的残差块类型 (BasicBlock 或 Bottleneck)。
            planes (int): 该层中残差块的基础通道数。
            blocks (int): 该层包含的残差块数量。
            stride (int, optional): 该层第一个残差块的步长，用于下采样。默认为 1。
            dilate (bool, optional): 是否用膨胀卷积替换步长。默认为 False。

        Returns:
            nn.Sequential: 由多个残差块组成的序列模块。
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 判断是否需要下采样连接 (downsample a.k.a. projection shortcut)
        # 条件是：步长不为1（需要下采样），或者输入通道数与块的输出通道数不匹配。
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 添加该层的第一块，它可能包含下采样
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        # 添加该层余下的块
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        定义 ResNet 的前向传播逻辑。

        Args:
            x (Tensor): 输入的图像张量。

        Returns:
            Tensor: 经过网络处理后的输出，通常是分类的 logits。
        """
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        包装 _forward_impl 以支持 TorchScript。
        """
        return self._forward_impl(x)


# resnet的某一个版本，即resnet50
def resnet_50(
        backbone_weights_state_dict: Optional[dict] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> ResNet:
    """
    方便地创建 ResNet-50 模型实例。

    用法:
        # 创建一个标准的 ResNet-50
        model = resnet_50()

        # 创建一个使用自定义 Normalization 层的 ResNet-50
        from functools import partial
        my_norm = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        model = resnet_50(norm_layer=my_norm)
        
        # 创建并加载预训练权重
        weights = torch.load("resnet50.pth")
        model = resnet_50(backbone_weights_state_dict=weights)

    Args:
        backbone_weights_state_dict (dict, optional): 预训练模型的权重字典。如果提供，将加载这些权重。默认为 None。
        norm_layer (Callable, optional): 要使用的归一化层。如果为 None，则使用 nn.BatchNorm2d。默认为 None。

    Returns:
        ResNet: 一个 ResNet-50 模型实例。
    """

    # 如果提供了权重，则根据权重中全连接层的输出维度来确定类别数
    if backbone_weights_state_dict is not None:
        num_classes = backbone_weights_state_dict["fc.weight"].shape[0]
    else:
        num_classes = 1000 # 默认为 ImageNet 的 1000 类

    # 使用 Bottleneck 块和 ResNet-50 的层配置 [3, 4, 6, 3] 来创建模型
    model = ResNet(
        block = Bottleneck, 
        layers = [3, 4, 6, 3], 
        num_classes=num_classes, 
        norm_layer=norm_layer)

    # 如果提供了权重，则加载
    if backbone_weights_state_dict is not None:
        model.load_state_dict(backbone_weights_state_dict)
    
    return model


# 特征金字塔网络构建
class BackboneWithFPN(nn.Module):
    """
    一个将骨干网络 (backbone) 和特征金字塔网络 (FPN) 结合在一起的包装模块。
    它的核心思想是：
    1. 使用 `IntermediateLayerGetter` 工具从骨干网络中提取出预先指定的中间层的输出。
    2. 将这些多尺度的中间层特征图送入 FPN 模块。
    3. FPN 对这些特征图进行融合和处理，生成一个具有丰富语义信息的多尺度特征金字塔。
    这个模块是构建现代目标检测模型（如 Faster R-CNN, Mask R-CNN）的基石。

    Args:
        backbone (nn.Module): 原始的骨干网络，例如一个 ResNet 实例。
        return_layers (Dict[str, str]): 一个字典，用于指定从 `backbone` 中提取哪些层的输出。
                                       key 是 `backbone` 中模块的名称 (如 'layer1')，
                                       value 是希望在输出字典中使用的名称 (如 'feat0')。
        in_channels_list (List[int]): 一个列表，按顺序包含 `return_layers` 中指定的每个特征图的通道数。
                                      这是 FPN 模块所必需的参数。
        out_channels (int): FPN 输出的所有特征图的统一通道数 (通常是 256)。
        extra_blocks (Optional[ExtraFPNBlock]): 在 FPN 之上添加的额外块，如 `LastLevelMaxPool`。
        norm_layer (Optional[Callable[..., nn.Module]]): 在 FPN 中使用的归一化层。
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        # self.body 实际上是一个 IntermediateLayerGetter 实例，它会返回 backbone 中间层的输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # 构建 FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        定义前向传播逻辑。

        Args:
            x (Tensor): 输入的图像张量。

        Returns:
            Dict[str, Tensor]: 一个有序字典，包含了 FPN 输出的多尺度特征图。
        """
        # 首先通过 self.body (IntermediateLayerGetter) 获取骨干网络的中间层输出
        x = self.body(x)
        # 然后将这些特征图送入 FPN 进行处理
        x = self.fpn(x)
        return x


# 给backbone添加fpn模式，即特征金字塔网络，进行特征融合
def resnet_fpn_extractor(
    backbone: ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    """
    一个辅助函数（工厂函数），用于从一个 ResNet 实例构建一个带有 FPN 的骨干网络。
    它封装了以下关键逻辑：
    1. 根据 `trainable_layers` 参数冻结 ResNet 的部分层。也就是是否选择预训练，或者自由选择训练哪些层。
    2. 确定要从 ResNet 的哪些层提取特征以输入 FPN。
    3. 自动计算这些特征图的通道数。
    4. 构建并返回一个 `BackboneWithFPN` 实例。

    Args:
        backbone (ResNet): 一个 ResNet 模型实例。
        trainable_layers (int): 一个范围在 [0, 5] 之间的整数，指定要解冻（训练）的 ResNet 层数。
                                0: 冻结所有层。
                                1: 只训练 'layer4'。
                                2: 训练 'layer3' 和 'layer4'。
                                ...
                                5: 训练所有层 ('conv1', 'bn1', 'layer1'...'layer4')。
        returned_layers (Optional[List[int]]): 一个列表，指定从 ResNet 的哪些 "stage"（层）返回特征图。
                                               有效值为 [1, 2, 3, 4]。默认为 `[1, 2, 3, 4]`。
        extra_blocks (Optional[ExtraFPNBlock]): 传递给 FPN 的额外块。
        norm_layer (Optional[Callable[..., nn.Module]]): 在 FPN 中使用的归一化层。

    Returns:
        BackboneWithFPN: 一个配置好的、带 FPN 的骨干网络。
    """

    # 根据 trainable_layers 冻结网络的某些层
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    
    # 从后往前选择要训练的层
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    # 如果要训练所有层，也包括第一个 BN 层
    if trainable_layers == 5:
        layers_to_train.append("bn1")
        
    # 遍历所有参数，如果其名称不是以任何一个要训练的层名开头，则将其 `requires_grad` 设为 False
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    
    # 构建 `return_layers` 字典，例如：{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    # `backbone.inplanes` 在 ResNet 初始化后是 64，这是 C2 之前的通道数。
    # 对于标准的 ResNet, layer1(C2) 的通道数是 256, layer2(C3)是512, ...
    # 这里 `in_channels_stage2` 应该是 layer1(C2) 的输出通道数，即 64 * 4 = 256
    # ResNet-50 中 layer1 的输出通道数是 256
    in_channels_stage2 = 256 
    # 计算 FPN 输入所需的通道数列表
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # FPN 输出的通道数固定为 256
    out_channels = 256
    
    # 创建并返回 BackboneWithFPN 实例
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )



