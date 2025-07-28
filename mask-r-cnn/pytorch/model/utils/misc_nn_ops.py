import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from model.utils.general import _make_ntuple


class FrozenBatchNorm2d(torch.nn.Module):
    """
    冻结的二维批量归一化 (Frozen Batch Normalization) 层。
    这个模块的行为与标准的 `nn.BatchNorm2d` 不同，它不更新训练期间的 `running_mean` 和 `running_var`，
    并且其 `weight` 和 `bias` 参数是固定的，不可训练的（注册为 buffer 而不是 Parameter）。
    它在训练期间总是使用加载的（或初始化的）统计数据进行归一化。

    这在微调从其他任务（如ImageNet分类）预训练的模型时非常有用，特别是当批次大小（batch size）
    很小的时候。使用小的批次大小会导致批量统计数据不稳定，冻结BN可以防止这种不稳定性破坏预训练的特征。

    用法:
        可以直接替换模型中的 `nn.BatchNorm2d` 层。

    Args:
        num_features (int): 输入特征图的通道数。
        eps (float, optional): 为防止除以零而加到方差上的一个很小的数。默认为 1e-5。
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        # 将 weight, bias, running_mean, running_var 注册为 buffer
        # buffer 是模型状态的一部分，会被保存到 state_dict 中，但不会被视为模型参数，因此不会被优化器更新。
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        # BatchNorm2d 在 PyTorch < 1.1.0 中有一个 'num_batches_tracked' 参数。
        # 为了向后兼容，在加载 state_dict 时，如果存在这个键，就将其移除。
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: Tensor) -> Tensor:
        # 手动执行批量归一化操作
        # (x - mean) / sqrt(var + eps) * weight + bias
        # 为了计算效率和对 TorchScript JIT 编译器的友好性，将所有参数 reshape
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        # 计算缩放和偏移量
        scale = w * (rv + self.eps).rsqrt()  # rsqrt()是求平方根的倒数
        bias = b - rm * scale
        # 应用归一化
        return x * scale + bias

    def __repr__(self) -> str:
        # 自定义模块的打印输出信息
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"


class ConvNormActivation(torch.nn.Sequential):
    """
    一个便利的模块，它将一个卷积层、一个归一化层和一个激活函数层按顺序组合在一起。
    这是一个通用的基类，不应该被直接使用。使用其特定维度的子类，如 `Conv2dNormActivation`。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int or tuple): 卷积核大小。
        stride (int or tuple, optional): 卷积步长。
        padding (int or tuple, optional): 填充大小。如果为None，会自动计算以保持空间维度。
        groups (int, optional): 分组卷积的组数。
        norm_layer (Callable[..., torch.nn.Module], optional): 归一化层。默认为 `torch.nn.BatchNorm2d`。
        activation_layer (Callable[..., torch.nn.Module], optional): 激活函数。默认为 `torch.nn.ReLU`。
        dilation (int or tuple, optional): 卷积核的膨胀率。
        inplace (bool, optional): 是否对激活函数使用原地操作 (in-place)。
        bias (bool, optional): 卷积层是否使用偏置。如果 `norm_layer` 存在，则通常设为 `False`。
        conv_layer (Callable[..., torch.nn.Module]): 使用的卷积层类型，如 `torch.nn.Conv2d`。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        # 如果没有指定 padding，则自动计算一个 "same" padding
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        
        # 如果 bias 未指定，则当存在 norm_layer 时，bias 设为 False，否则为 True
        # 这是因为 BatchNorm 层本身有可学习的偏移参数，卷积层的 bias 是多余的。
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
            
        # 调用父类 nn.Sequential 的构造函数
        super().__init__(*layers)
        self.out_channels = out_channels

        # 如果直接实例化这个基类，则发出警告
        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    一个便利的模块，它将一个 `Conv2d`、一个归一化层（默认为`BatchNorm2d`）和一个激活函数（默认为`ReLU`）组合在一起。
    这是 `ConvNormActivation` 的二维版本。

    用法:
        # 创建一个 Conv-BN-ReLU 块
        conv_block = Conv2dNormActivation(3, 64, kernel_size=3, stride=1)
        # 创建一个 Conv-IN-LeakyReLU 块
        from functools import partial
        conv_block_2 = Conv2dNormActivation(3, 64,
                                            norm_layer=partial(nn.InstanceNorm2d, affine=True),
                                            activation_layer=nn.LeakyReLU)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        # 调用父类的构造函数，并明确指定使用 torch.nn.Conv2d
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

