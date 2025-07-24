import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms.functional as F
from transforms.mask_rcnn_transforms import Compose
from transforms.mask_rcnn_transforms import build_mask_rcnn_transforms


class COCODataset(Dataset):
    """
    标准 COCO 格式数据集加载器，适用于目标检测 + 实例分割（Mask R-CNN）。
    """

    def __init__(self, image_dir, ann_file, transforms=None, train=True):
        """
        Args:
            image_dir (str): 图像目录路径，如 '.../train2017'
            ann_file (str): 标注 JSON 文件路径，如 '.../annotations/instances_train2017.json'
            transforms (callable, optional): 图像与 target 的联合变换
            train (bool): 是否为训练模式，影响数据增强策略
        """
        self.root = image_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.train = train
        
        # 检查和打印数据集信息
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self.classes)
        
        # 创建类别 ID 到连续 ID 的映射
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_continuous_id = {
            coco_id: i + 1  # 背景为0
            for i, coco_id in enumerate(self.cat_ids)
        }
        
        print(f"加载了 {len(self.ids)} 张图像，{len(self.cat_ids)} 个类别")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        # 读取图像
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        
        # 获取原始图像尺寸
        width, height = img.size

        # 解析目标实例
        boxes = []
        labels = []
        masks = []
        iscrowd = []
        area = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue  # 跳过 crowd 区域

            # 检查 bbox 和 segmentation 是否存在
            if len(ann['bbox']) == 0 or len(ann['segmentation']) == 0:
                continue
                
            # 将COCO类别ID映射到连续的类别ID
            cat_id = ann['category_id']
            class_id = self.cat_id_to_continuous_id[cat_id]

            # bbox: [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            
            # 边界检查，确保边界框在图像内部
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            
            # 筛选出太小的框
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
                
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            mask = coco.annToMask(ann)
            masks.append(mask)
            iscrowd.append(ann.get('iscrowd', 0))
            area.append(ann.get('area', w * h))

        # 如果没有标注，返回一个简单的样本
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

        image_id = torch.tensor([img_id])

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([int(height), int(width)]),
        }

        # 应用变换
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            # 基本转换
            img = F.to_tensor(img)  # 转成 FloatTensor，[0,1]

        return img, target

    def __len__(self):
        return len(self.ids)


def get_coco_api_from_dataset(dataset):
    """
    从数据集获取 COCO API 对象
    """
    return dataset.coco


def build_coco_dataset(image_dir, ann_file, transforms=None, train=True):
    """
    构建 COCO 数据集
    """
    return COCODataset(image_dir, ann_file, transforms, train)


def build_dataloader(dataset, batch_size=2, num_workers=4, train=True):
    """
    构建 DataLoader
    """
    from transforms.mask_rcnn_transforms import collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def create_coco_dataloader(config, train=True):
    """
    根据配置创建 COCO 数据集和 DataLoader
    
    Args:
        config: 配置对象，包含数据集和训练参数
        train: 是否为训练模式
    
    Returns:
        dataset, dataloader: 数据集和数据加载器
    """
    # 获取路径配置
    root_dir = config['dataset']['root_dir']
    
    if train:
        image_dir = os.path.join(root_dir, config['dataset']['train_images'])
        ann_file = os.path.join(root_dir, config['dataset']['train_ann'])
    else:
        image_dir = os.path.join(root_dir, config['dataset']['val_images'])
        ann_file = os.path.join(root_dir, config['dataset']['val_ann'])
    
    # 构建适合Mask R-CNN的数据变换
    transforms = build_mask_rcnn_transforms(
        train=train,
        min_size=config['transforms']['resize_min'], 
        max_size=config['transforms']['resize_max']
    )
    
    # 构建数据集
    dataset = build_coco_dataset(image_dir, ann_file, transforms, train)
    
    # 构建数据加载器
    batch_size = config['training']['batch_size'] if train else 1
    num_workers = config['training']['workers']
    
    dataloader = build_dataloader(dataset, batch_size, num_workers, train)
    
    return dataset, dataloader
