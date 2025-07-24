"""
适合Mask R-CNN的数据增强方法
包含：
1. LSJ (Large Scale Jitter) - 大尺度抖动
2. RandomHorizontalFlip - 随机水平翻转
3. ColorJitter - 颜色抖动
4. Grayscale - 随机灰度化
5. 小角度旋转
6. 小幅度裁剪（确保实例完整性）
7. Normalize - 标准化
"""

import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageOps

class Compose:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class LargeScaleJitter:
    """
    大尺度抖动 (LSJ)
    参考论文：https://arxiv.org/abs/2106.05237
    """
    def __init__(self, min_scale=0.1, max_scale=2.0, prob=1.0):
        """
        参数:
            min_scale (float): 最小缩放比例
            max_scale (float): 最大缩放比例
            prob (float): 应用此变换的概率
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # 获取原始尺寸
        orig_width, orig_height = image.size
        
        # 随机选择缩放因子
        scale_factor = random.uniform(self.min_scale, self.max_scale)
        
        # 新的尺寸
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # 缩放图像
        image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # 缩放目标
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'] * scale_factor
            target['boxes'] = boxes
            
            # 更新面积
            if 'area' in target:
                target['area'] = target['area'] * (scale_factor * scale_factor)
        
        # 缩放掩码
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks']
            new_masks = []
            
            for mask in masks:
                mask_pil = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)
                mask_np = np.array(mask_pil) > 0
                new_masks.append(torch.tensor(mask_np, dtype=torch.uint8))
            
            if new_masks:
                target['masks'] = torch.stack(new_masks)
        
        return image, target

class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            width = image.size[0]

            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # flip x1 <-> x2
                target['boxes'] = boxes

            if 'masks' in target and len(target['masks']) > 0:
                target['masks'] = target['masks'].flip(-1)  # 水平翻转掩码

        return image, target

class ColorJitterTransform:
    """颜色抖动变换"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.8):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.color_jitter(image)
        return image, target

class RandomGrayscale:
    """随机灰度化"""
    def __init__(self, prob=0.1):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = ImageOps.grayscale(image)
            # 转换为3通道灰度图
            image = Image.merge("RGB", [image, image, image])
        return image, target

class SmallRotation:
    """小角度旋转，保持实例完整性"""
    def __init__(self, angle_range=10, prob=0.3, expand=False):
        self.angle_range = angle_range
        self.prob = prob
        self.expand = expand  # 是否扩展图像以包含整个旋转后的图像
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # 随机选择角度
        angle = random.uniform(-self.angle_range, self.angle_range)
        
        # 旋转图像
        rotated_image = image.rotate(angle, expand=self.expand, resample=Image.BILINEAR)
        
        # 处理目标数据
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks']
            rotated_masks = []
            
            # 旋转每个掩码
            for mask in masks:
                mask_pil = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                rotated_mask = mask_pil.rotate(angle, expand=self.expand, resample=Image.NEAREST)
                rotated_mask_np = np.array(rotated_mask) > 0
                rotated_masks.append(torch.tensor(rotated_mask_np, dtype=torch.uint8))
            
            if rotated_masks:
                target['masks'] = torch.stack(rotated_masks)
                
                # 从掩码重新计算边界框
                if 'boxes' in target:
                    boxes = []
                    labels = []
                    
                    for i, mask in enumerate(target['masks']):
                        pos = torch.where(mask)
                        if len(pos[0]) > 0 and len(pos[1]) > 0:
                            y1, y2 = pos[0].min(), pos[0].max()
                            x1, x2 = pos[1].min(), pos[1].max()
                            
                            # 确保框有效（不是单点）
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                                labels.append(target['labels'][i].item())
                    
                    if boxes:
                        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                        target['labels'] = torch.tensor(labels, dtype=torch.int64)
                        # 更新面积
                        if 'area' in target:
                            target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        
        return rotated_image, target

class SafeRandomCrop:
    """安全随机裁剪，保证实例的完整性"""
    def __init__(self, max_crop_fraction=0.2, min_instance_area=0.8, prob=0.3):
        """
        参数:
            max_crop_fraction: 最大裁剪比例 (相对于原始尺寸)
            min_instance_area: 保留的最小实例面积比例
            prob: 应用裁剪的概率
        """
        self.max_crop_fraction = max_crop_fraction
        self.min_instance_area = min_instance_area
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        width, height = image.size
        
        # 计算最大裁剪量
        max_crop_x = int(width * self.max_crop_fraction)
        max_crop_y = int(height * self.max_crop_fraction)
        
        # 随机选择裁剪量
        crop_left = random.randint(0, max_crop_x)
        crop_top = random.randint(0, max_crop_y)
        crop_right = random.randint(0, max_crop_x)
        crop_bottom = random.randint(0, max_crop_y)
        
        # 计算新的裁剪区域
        new_left = crop_left
        new_top = crop_top
        new_right = width - crop_right
        new_bottom = height - crop_bottom
        
        # 确保裁剪区域有效
        if new_left >= new_right or new_top >= new_bottom:
            return image, target
        
        # 检查实例是否保持完整性
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            
            # 计算每个实例与裁剪区域的重叠部分
            keep_indices = []
            new_boxes = []
            
            for i, box in enumerate(boxes):
                # 计算交集
                ix1 = max(box[0], new_left)
                iy1 = max(box[1], new_top)
                ix2 = min(box[2], new_right)
                iy2 = min(box[3], new_bottom)
                
                if ix2 > ix1 and iy2 > iy1:
                    # 计算交集面积比例
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    overlap_area = (ix2 - ix1) * (iy2 - iy1)
                    ratio = overlap_area / box_area
                    
                    if ratio >= self.min_instance_area:
                        # 保留此实例
                        keep_indices.append(i)
                        # 更新边界框坐标
                        new_boxes.append([
                            max(ix1 - new_left, 0),
                            max(iy1 - new_top, 0),
                            min(ix2 - new_left, new_right - new_left),
                            min(iy2 - new_top, new_bottom - new_top)
                        ])
            
            # 如果没有保留任何实例，放弃裁剪
            if not keep_indices:
                return image, target
            
            # 执行裁剪
            cropped_image = image.crop((new_left, new_top, new_right, new_bottom))
            
            # 更新标签
            target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
            target['labels'] = target['labels'][keep_indices]
            
            # 裁剪掩码
            if 'masks' in target:
                masks = target['masks']
                new_masks = []
                
                for i in keep_indices:
                    mask = masks[i]
                    cropped_mask = mask[new_top:new_bottom, new_left:new_right]
                    new_masks.append(cropped_mask)
                
                if new_masks:
                    target['masks'] = torch.stack(new_masks)
                
                # 更新面积
                if 'area' in target:
                    target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
            
            return cropped_image, target
        
        # 如果没有边界框，直接裁剪
        cropped_image = image.crop((new_left, new_top, new_right, new_bottom))
        return cropped_image, target

class Normalize:
    """标准化图像"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        # 将PIL图像转换为tensor并标准化
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ToTensor:
    """将PIL图像转换为tensor"""
    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        return image, target

class Resize:
    """调整图像大小"""
    def __init__(self, min_size=800, max_size=1333):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, image, target):
        # 获取原始尺寸
        w, h = image.size
        
        # 计算缩放比例
        scale = self.min_size / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)
        
        # 新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整图像大小
        image = F.resize(image, (new_h, new_w))
        
        # 调整边界框
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'] * scale
            target['boxes'] = boxes
            
            # 更新面积
            if 'area' in target:
                target['area'] = target['area'] * (scale * scale)
        
        # 调整掩码
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks'].unsqueeze(1).float()  # [N, 1, H, W]
            masks = torch.nn.functional.interpolate(masks, size=(new_h, new_w), mode='nearest')
            target['masks'] = masks.squeeze(1).byte()
        
        return image, target

def build_mask_rcnn_transforms(train=True, min_size=800, max_size=1333):
    """
    构建适合Mask R-CNN的数据变换流程
    
    参数:
        train: 是否为训练模式
        min_size: 最小尺寸
        max_size: 最大尺寸
    
    返回:
        transforms: 变换流程
    """
    transforms_list = []
    
    if train:
        # 训练模式下添加数据增强
        # 1. 大尺度抖动 (LSJ)
        transforms_list.append(LargeScaleJitter(min_scale=0.3, max_scale=2.0, prob=0.5))
        
        # 2. 随机水平翻转
        transforms_list.append(RandomHorizontalFlip(prob=0.5))
        
        # 3. 小角度旋转 (±10度)
        transforms_list.append(SmallRotation(angle_range=10, prob=0.3))
        
        # 4. 安全随机裁剪
        transforms_list.append(SafeRandomCrop(max_crop_fraction=0.2, min_instance_area=0.8, prob=0.3))
        
        # 5. 颜色抖动
        transforms_list.append(ColorJitterTransform(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.8))
        
        # 6. 随机灰度化
        transforms_list.append(RandomGrayscale(prob=0.1))
    
    # 添加标准调整大小（训练和验证都需要）
    transforms_list.append(Resize(min_size=min_size, max_size=max_size))
    
    # 添加标准化和转tensor（训练和验证都需要）
    transforms_list.append(ToTensor())
    transforms_list.append(Normalize())
    
    return Compose(transforms_list)

def collate_fn(batch):
    """
    数据批处理函数
    """
    return tuple(zip(*batch)) 