# 导入必要的库
import os  # 导入操作系统相关的功能模块
import torch  # 导入PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器类
from torchvision.io import read_image  # 导入图像读取函数
import numpy as np  # 导入数值计算库
from pycocotools.coco import COCO  # 导入COCO数据集工具
import torchvision.transforms.functional as F  # 导入图像变换函数
from PIL import Image  # 导入PIL图像处理库
from transforms.mask_rcnn_transforms import Compose, build_mask_rcnn_transforms  # 导入自定义的Mask R-CNN数据变换


class COCODataset(Dataset):
    """
    标准 COCO 格式数据集加载器，继承于torch.utils.data.Dataset。
    主要用于加载COCO数据集，并进行数据增强。
    
    参数：
    - image_dir: 图像目录
    - ann_file: 标注文件
    - transforms: 数据增强
    """

    def __init__(self, image_dir, ann_file, transforms=None, train=True):
        """
        初始化COCO数据集
        
        参数:
        - image_dir: 图像文件夹路径
        - ann_file: 标注文件路径
        - transforms: 数据增强转换
        - train: 是否为训练模式
        """
        self.root = image_dir  # 存储图像目录路径
        self.coco = COCO(ann_file)  # 使用COCO API加载标注文件
        self.ids = list(sorted(self.coco.imgs.keys()))  # 获取并排序所有图像ID
        self.transforms = transforms  # 存储数据增强转换
        self.train = train  # 存储是否为训练模式

        # 类别信息
        cats = self.coco.dataset['categories']  # 不使用 getCatIds()，直接从数据集中获取类别信息
        # 构建 COCO id 到 class name 的完整映射（按类别 id 升序排序）
        cats = sorted(cats, key=lambda x: x['id'])  # 按类别ID排序
        self.classes = tuple(['__background__'] + [c['name'] for c in cats])  # 创建类别名称列表，添加背景类
        self.num_classes = len(self.classes)  # 计算类别总数
        self.cat_ids = [c['id'] for c in cats]  # 获取COCO原始类别ID列表，例如 [1, 2, 3, 4, 5, ..., 90, 91]
        # 建立 COCO id → 连续 id（1~91）映射
        self.cat_id_to_continuous_id = {coco_id: idx + 1 for idx, coco_id in enumerate(self.cat_ids)}  # 创建COCO ID到连续ID的映射字典


        print(f"加载了 {len(self.ids)} 张图像，{len(self.cat_ids)} 个类别")  # 打印加载的图像数量和类别数量

    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        
        参数:
        - index: 样本索引
        
        返回:
        - img: 处理后的图像张量
        - target: 包含标注信息的字典
        """
        coco = self.coco  # 获取COCO API对象
        img_id = self.ids[index]  # 获取当前索引对应的图像ID
        ann_ids = coco.getAnnIds(imgIds=img_id)  # 获取该图像的所有标注ID
        anns = coco.loadAnns(ann_ids)  # 加载所有标注信息
        img_info = coco.loadImgs(img_id)[0]  # 加载图像信息
        path = img_info['file_name']  # 获取图像文件名

        # 使用 PIL 加载图像，并且转化为tensor
        img_path = os.path.join(self.root, path)  # 构建完整的图像路径
        img = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB模式
        img = F.pil_to_tensor(img).float() / 255.0  # 将PIL图像转换为张量并归一化

        # 获取原始尺寸（从 COCO 元数据）
        width = img_info['width']  # 获取图像宽度
        height = img_info['height']  # 获取图像高度

        boxes = []  # 初始化边界框列表
        labels = []  # 初始化标签列表
        masks = []  # 初始化掩码列表
        iscrowd = []  # 初始化是否为群体标志列表
        area = []  # 初始化区域面积列表

        for ann in anns:  # 遍历所有标注
            if ann.get('iscrowd', 0):  # 跳过群体标注
                continue
            if len(ann['bbox']) == 0 or len(ann['segmentation']) == 0:  # 跳过没有边界框或分割信息的标注
                continue

            cat_id = ann['category_id']  # 获取类别ID
            class_id = self.cat_id_to_continuous_id[cat_id]  # 将COCO类别ID转换为连续类别ID

            x, y, w, h = ann['bbox']  # 获取边界框坐标（x, y, 宽, 高）
            x1 = max(0, x)  # 确保x1不小于0
            y1 = max(0, y)  # 确保y1不小于0
            x2 = min(width, x + w)  # 确保x2不超过图像宽度
            y2 = min(height, y + h)  # 确保y2不超过图像高度

            if x2 - x1 < 1 or y2 - y1 < 1:  # 跳过太小的边界框
                continue

            boxes.append([x1, y1, x2, y2])  # 添加边界框坐标
            labels.append(class_id)  # 添加类别ID
            mask = coco.annToMask(ann)  # 将标注转换为二值掩码
            masks.append(mask)  # 添加掩码
            iscrowd.append(ann.get('iscrowd', 0))  # 添加是否为群体标志
            area.append(ann.get('area', w * h))  # 添加区域面积，如果没有则计算

        if len(boxes) == 0:  # 如果没有有效的标注
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # 创建空的边界框张量
            labels = torch.zeros((0,), dtype=torch.int64)  # 创建空的标签张量
            masks = torch.zeros((0, height, width), dtype=torch.uint8)  # 创建空的掩码张量
            area = torch.zeros((0,), dtype=torch.float32)  # 创建空的面积张量
            iscrowd = torch.zeros((0,), dtype=torch.uint8)  # 创建空的群体标志张量
        else:  # 如果有有效的标注
            boxes = torch.as_tensor(boxes, dtype=torch.float32)  # 将边界框列表转换为张量
            labels = torch.as_tensor(labels, dtype=torch.int64)  # 将标签列表转换为张量
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)  # 将掩码列表堆叠并转换为张量
            area = torch.as_tensor(area, dtype=torch.float32)  # 将面积列表转换为张量
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)  # 将群体标志列表转换为张量

        image_id = torch.tensor([img_id])  # 创建图像ID张量

        target = {  # 创建目标字典
            'boxes': boxes,  # 边界框
            'labels': labels,  # 类别标签
            'masks': masks,  # 实例分割掩码
            'image_id': image_id,  # 图像ID
            'area': area,  # 区域面积
            'iscrowd': iscrowd,  # 是否为群体
            'orig_size': torch.as_tensor([int(height), int(width)]),  # 原始图像尺寸
        }

        if self.transforms is not None:  # 如果有数据增强转换
            img, target = self.transforms(img, target)  # 应用数据增强

        return img, target  # 返回图像和目标

    def __len__(self):
        """
        返回数据集中样本的数量
        
        返回:
        - 数据集大小
        """
        return len(self.ids)  # 返回图像ID列表的长度


def get_coco_api_from_dataset(dataset):
    """
    从数据集对象获取COCO API对象
    
    参数:
    - dataset: COCO数据集对象
    
    返回:
    - COCO API对象
    """
    return dataset.coco  # 返回数据集的COCO对象


def build_coco_dataset(image_dir, ann_file, transforms=None, train=True):
    """
    构建COCO数据集
    
    参数:
    - image_dir: 图像目录
    - ann_file: 标注文件
    - transforms: 数据增强
    - train: 是否为训练模式
    
    返回:
    - COCO数据集对象
    """
    return COCODataset(image_dir, ann_file, transforms, train)  # 创建并返回COCODataset实例


def build_dataloader(dataset, batch_size=4, num_workers=4, train=True):
    """
    构建数据加载器
    
    参数:
    - dataset: 数据集对象
    - batch_size: 批量大小
    - num_workers: 工作进程数
    - train: 是否为训练模式
    
    返回:
    - 数据加载器对象
    """
    from transforms.mask_rcnn_transforms import collate_fn  # 导入自定义的收集函数
    return DataLoader(  # 创建并返回DataLoader实例
        dataset,  # 数据集
        batch_size=batch_size,  # 批量大小
        shuffle=train,  # 是否打乱数据，训练时打乱
        num_workers=num_workers,  # 工作进程数
        collate_fn=collate_fn,  # 自定义收集函数
        pin_memory=True,  # 使用固定内存，加速数据传输到GPU
    )


def create_coco_dataloader(config, train=True):
    """
    根据配置创建COCO数据加载器
    
    参数:
    - config: 配置字典
    - train: 是否为训练模式
    
    返回:
    - dataset: 数据集对象
    - dataloader: 数据加载器对象
    """
    root_dir = config['dataset']['root_dir']  # 获取数据集根目录
    if train:  # 如果是训练模式
        image_dir = os.path.join(root_dir, config['dataset']['train_images'])  # 获取训练图像目录
        ann_file = os.path.join(root_dir, config['dataset']['train_ann'])  # 获取训练标注文件
    else:  # 如果是验证模式
        image_dir = os.path.join(root_dir, config['dataset']['val_images'])  # 获取验证图像目录
        ann_file = os.path.join(root_dir, config['dataset']['val_ann'])  # 获取验证标注文件

    # 获取数据增强级别
    augmentation_level = config['transforms'].get('augmentation_level', 2)  # 获取数据增强级别，默认为2

    transforms = build_mask_rcnn_transforms(  # 构建Mask R-CNN数据转换
        train=train,  # 是否为训练模式
        min_size=config['transforms']['resize_min'],  # 最小尺寸
        max_size=config['transforms']['resize_max'],  # 最大尺寸
        augmentation_level=augmentation_level  # 数据增强级别
    )

    dataset = build_coco_dataset(image_dir, ann_file, transforms, train)  # 构建COCO数据集
    batch_size = config['training']['batch_size'] if train else 1  # 获取批量大小，验证时为1
    num_workers = config['training']['workers']  # 获取工作进程数

    dataloader = build_dataloader(dataset, batch_size, num_workers, train)  # 构建数据加载器
    return dataset, dataloader  # 返回数据集和数据加载器