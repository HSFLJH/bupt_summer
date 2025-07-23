# 数据集加载模块 - COCO数据集处理类
import os
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

class CocoInstanceDataset(CocoDetection):
    """
    COCO实例分割数据集类
    继承自torchvision的CocoDetection，专门用于Mask R-CNN训练
    
    功能：
    1. 加载COCO格式的图像和标注
    2. 提取边界框、类别标签和分割mask
    3. 转换为PyTorch张量格式
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        """
        初始化数据集
        
        参数:
            img_folder: 图像文件夹路径 (如: coco/train2017)
            ann_file: 标注文件路径 (如: coco/annotations/instances_train2017.json)
            transforms: 图像变换函数 (可选)
        """
        super(CocoInstanceDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        这是数据加载的核心函数，完成以下步骤：
        1. 加载原始图像和标注
        2. 提取边界框坐标
        3. 提取类别标签
        4. 生成分割mask
        5. 转换为PyTorch张量格式
        
        返回: (图像张量, 目标字典)
        """
        # 1. 【数据加载】从COCO数据集获取图像和标注
        img, target = super(CocoInstanceDataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        
        # 2. 【标注解析】获取该图像的所有标注ID和详细标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 3. 【数据预处理】提取边界框、标签和mask
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            if 'bbox' in ann:
                # 【边界框处理】将COCO格式[x,y,w,h]转换为[x1,y1,x2,y2]
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                
                # 【标签处理】提取类别ID
                labels.append(ann['category_id'])
                
                # 【mask生成】将COCO多边形标注转换为二值mask
                mask = self.coco.annToMask(ann)
                masks.append(mask)

        # 4. 【张量转换】将列表转换为PyTorch张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # 5. 【目标字典组装】按照Mask R-CNN要求的格式组织数据
        target = {
            "boxes": boxes,        # 边界框 [N, 4]
            "labels": labels,      # 类别标签 [N]
            "masks": masks,        # 分割mask [N, H, W]
            "image_id": torch.tensor([img_id])  # 图像ID
        }

        # 6. 【图像预处理】应用图像变换（如果有的话）
        if self._transforms:
            img = self._transforms(img)

        return img, target

    def __len__(self):
        """返回数据集大小"""
        return len(self.ids)


def get_transform(train=True):
    """
    获取图像预处理变换
    
    参数:
        train: 是否为训练模式
        
    返回:
        训练模式: 包含数据增强的变换组合
        验证模式: 仅包含基础预处理的变换
    """
    import torchvision.transforms as T
    
    if train:
        # 【训练时数据增强】
        return T.Compose([
            T.ToTensor(),                    # 转换为张量并归一化到[0,1]
            T.RandomHorizontalFlip(0.5),     # 50%概率水平翻转
        ])
    else:
        # 【验证时基础预处理】
        return T.Compose([
            T.ToTensor(),                    # 仅转换为张量
        ])
