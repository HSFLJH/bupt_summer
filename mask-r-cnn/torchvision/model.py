# 模型定义模块 - Mask R-CNN模型构建
# 基于torchvision预训练模型构建自定义的Mask R-CNN
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    """
    构建实例分割模型（Mask R-CNN）
    
    参数:
        num_classes: 类别数量（包括背景类）
        
    返回:
        配置好的Mask R-CNN模型
    
    模型结构:
    1. 骨干网络: ResNet50 + FPN (特征金字塔网络)
    2. RPN: 区域建议网络
    3. ROI Head: 包含分类、回归和mask分支
    """
    # 【基础模型加载】加载预训练的Mask R-CNN模型
    # pretrained=True: 使用在COCO数据集上预训练的权重
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
        num_classes = num_classes)

    # 【分类头替换】替换ROI分类器以适应自定义类别数
    # 获取分类器的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 创建新的分类器：输入特征维度 -> 自定义类别数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 【mask头替换】替换mask预测器以适应自定义类别数
    # 获取mask预测器的输入通道数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # 隐藏层维度
    # 创建新的mask预测器：输入维度 -> 隐藏层 -> 自定义类别数
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
