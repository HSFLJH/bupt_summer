from model.utils.misc_nn_ops import FrozenBatchNorm2d
from torch import nn
from model.backbone.resnet50 import resnet_50, resnet_fpn_extractor
import torch
from model.rcnn.mask_rcnn import MaskRCNN

def get_model_mask_r_cnn(
        is_trained: bool = True,
        num_classes: int = 91,
        progess: bool = True, # 是否显示下载预训练参数的进度条
        trainable_backbone_layers: int = 3,
) -> MaskRCNN:
    """
    
    """
    
    # ============================== 初始化各种参数 ==============================

    if is_trained:
        trainable_backbone_layers = 3 # 如果预训练，则默认使用3层可训练层
        norm_layer = FrozenBatchNorm2d if is_trained else nn.BatchNorm2d # 如果预训练，则使用FrozenBatchNorm2d，否则使用BatchNorm2d
        # 这个是冻结的批量归一化，它永远不会从当前的小批量数据中计算 mean 和 variance，使得噪声变小，训练更稳定

    # ============================== 加载预训练权重 ==============================

    if is_trained:
        maskrcnn_weights_url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        backbone_weights_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"

        maskrcnn_weights_state_dict = torch.hub.load_state_dict_from_url(maskrcnn_weights_url, progress=progess, check_hash=True)
        backbone_weights_state_dict = torch.hub.load_state_dict_from_url(backbone_weights_url, progress=progess, check_hash=True)
    else:
        maskrcnn_weights_state_dict = None
        backbone_weights_state_dict = None

    # ============================== 初始化骨干网络 ==============================

        # 带有参数的backbone:ResNet50
    backbone = resnet_50(backbone_weights_state_dict = backbone_weights_state_dict, norm_layer = norm_layer)
        # 给backbone添加fpn模式，即特征金字塔网络，进行特征融合
    backbone = resnet_fpn_extractor(backbone, trainable_backbone_layers)

    # ============================== 初始化RCNN网络 ==============================

    model = MaskRCNN(backbone, num_classes = num_classes)
    model.load_state_dict(maskrcnn_weights_state_dict)

    return model

    





