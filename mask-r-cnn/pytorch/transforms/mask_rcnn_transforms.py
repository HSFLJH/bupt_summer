import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from typing import Dict, List, Optional, Tuple


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class LargeScaleJitter:
    def __init__(self, min_scale=0.1, max_scale=2.0, prob=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        _, h, w = image.shape
        scale = random.uniform(self.min_scale, self.max_scale)
        nh, nw = int(h * scale), int(w * scale)

        image = F.resize(image, [nh, nw])

        if 'boxes' in target:
            target['boxes'] = target['boxes'] * scale
        if 'area' in target:
            target['area'] = target['area'] * (scale ** 2)
        if 'masks' in target:
            masks = target['masks'].unsqueeze(1).float()
            masks = torch.nn.functional.interpolate(masks, size=(nh, nw), mode='nearest')
            target['masks'] = masks.squeeze(1).byte()

        return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        return image, target

class ColorJitterTransform:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.8):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.jitter(image)
        return image, target

class RandomGrayscale:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = T.RandomGrayscale(p=1.0)(image)
        return image, target

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Resize:
    def __init__(self, min_size=800, max_size=1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        _, h, w = image.shape
        scale = self.min_size / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)

        nh, nw = int(h * scale), int(w * scale)
        image = F.resize(image, [nh, nw])

        if 'boxes' in target:
            target['boxes'] = target['boxes'] * scale
        if 'area' in target:
            target['area'] = target['area'] * (scale ** 2)
        if 'masks' in target:
            masks = target['masks'].unsqueeze(1).float()
            masks = torch.nn.functional.interpolate(masks, size=(nh, nw), mode='nearest')
            target['masks'] = masks.squeeze(1).byte()

        return image, target

class ToTensor:
    def __call__(self, image, target):
        return image, target

class SmallRotation:
    """小角度旋转，保持实例完整性（纯 tensor 版本）"""
    def __init__(self, angle_range=10, prob=0.3, expand=False):
        self.angle_range = angle_range
        self.prob = prob
        self.expand = expand  # 是否扩展图像尺寸以包含完整旋转图
        self.interpolation_image = F.InterpolationMode.BILINEAR
        self.interpolation_mask = F.InterpolationMode.NEAREST

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        angle = random.uniform(-self.angle_range, self.angle_range)

        # Tensor 上旋转图像
        image = F.rotate(image, angle, interpolation=self.interpolation_image, expand=self.expand)

        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks']
            rotated_masks = []
            for mask in masks:
                rotated = F.rotate(mask.unsqueeze(0), angle, interpolation=self.interpolation_mask, expand=self.expand)
                rotated_masks.append(rotated.squeeze(0))
            target['masks'] = torch.stack(rotated_masks)

            # 从掩码重新计算 boxes
            if 'boxes' in target:
                boxes = []
                labels = []
                areas = []
                for i, mask in enumerate(target['masks']):
                    pos = torch.where(mask)
                    if len(pos[0]) > 0 and len(pos[1]) > 0:
                        y1, y2 = pos[0].min().item(), pos[0].max().item()
                        x1, x2 = pos[1].min().item(), pos[1].max().item()
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(target['labels'][i].item())
                            areas.append((y2 - y1) * (x2 - x1))
                if boxes:
                    target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                    target['labels'] = torch.tensor(labels, dtype=torch.int64)
                    if 'area' in target:
                        target['area'] = torch.tensor(areas, dtype=torch.float32)

        return image, target

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
        
        # 获取图像尺寸
        _, h, w = image.shape
        
        # 计算最大裁剪量
        max_crop_x = int(w * self.max_crop_fraction)
        max_crop_y = int(h * self.max_crop_fraction)
        
        # 随机选择裁剪量
        crop_left = random.randint(0, max_crop_x)
        crop_top = random.randint(0, max_crop_y)
        crop_right = random.randint(0, max_crop_x)
        crop_bottom = random.randint(0, max_crop_y)
        
        # 计算新的裁剪区域
        new_left = crop_left
        new_top = crop_top
        new_right = w - crop_right
        new_bottom = h - crop_bottom
        
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
            cropped_image = image[:, new_top:new_bottom, new_left:new_right]
            
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
        cropped_image = image[:, new_top:new_bottom, new_left:new_right]
        return cropped_image, target

class MotionBlur:
    """运动模糊，模拟相机运动或物体运动"""
    def __init__(self, kernel_size=7, angle_range=180, prob=0.3):
        """
        参数:
            kernel_size: 模糊核大小
            angle_range: 模糊角度范围(0-180度)
            prob: 应用模糊的概率
        """
        self.kernel_size = kernel_size
        self.angle_range = angle_range
        self.prob = prob
    
    def _motion_blur_kernel(self, kernel_size, angle):
        """创建运动模糊核"""
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)
        
        # 计算直线的方向向量
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # 在核中绘制一条线
        for i in range(kernel_size):
            offset = i - center
            x = center + dx * offset
            y = center + dy * offset
            
            # 检查点是否在核内
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                # 双线性插值
                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, kernel_size - 1), min(y0 + 1, kernel_size - 1)
                
                # 计算权重
                wx = x - x0
                wy = y - y0
                
                # 设置值
                kernel[y0, x0] += (1 - wx) * (1 - wy)
                kernel[y0, x1] += wx * (1 - wy)
                kernel[y1, x0] += (1 - wx) * wy
                kernel[y1, x1] += wx * wy
        
        # 归一化核
        kernel = kernel / kernel.sum()
        return kernel
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # 随机选择角度
        angle = random.uniform(0, self.angle_range)
        
        # 创建模糊核
        kernel = self._motion_blur_kernel(self.kernel_size, angle)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        
        # 应用模糊
        # 为每个通道创建卷积核
        kernel = kernel.repeat(3, 1, 1, 1)
        
        # 使用卷积应用模糊
        padding = self.kernel_size // 2
        blurred_image = torch.nn.functional.conv2d(
            image.unsqueeze(0), 
            kernel, 
            padding=padding, 
            groups=3
        ).squeeze(0)
        
        # 确保图像数值在有效范围内
        blurred_image = torch.clamp(blurred_image, 0, 1)
        
        return blurred_image, target

class RandomPerspective:
    """随机透视变换，适用于 tensor 图像"""
    def __init__(self, distortion_scale=0.2, prob=0.3):
        self.distortion_scale = distortion_scale
        self.prob = prob
        self.interpolation_image = F.InterpolationMode.BILINEAR
        self.interpolation_mask = F.InterpolationMode.NEAREST

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        _, h, w = image.shape

        # 定义 startpoints：原图四个角
        startpoints = [
            [0, 0],             # top-left
            [w - 1, 0],         # top-right
            [w - 1, h - 1],     # bottom-right
            [0, h - 1]          # bottom-left
        ]

        # 计算 endpoints：在 startpoints 附近进行扰动
        def distort(pt):
            dx = random.uniform(-self.distortion_scale, self.distortion_scale) * w
            dy = random.uniform(-self.distortion_scale, self.distortion_scale) * h
            return [pt[0] + dx, pt[1] + dy]

        endpoints = [distort(pt) for pt in startpoints]

        # 图像透视变换
        image = F.perspective(image, startpoints, endpoints, interpolation=self.interpolation_image)

        # mask 透视变换
        if 'masks' in target and len(target['masks']) > 0:
            new_masks = []
            for mask in target['masks']:
                mask = F.perspective(mask.unsqueeze(0), startpoints, endpoints, interpolation=self.interpolation_mask)
                new_masks.append(mask.squeeze(0))
            target['masks'] = torch.stack(new_masks)

            # 从掩码重建 box、labels、area
            if 'boxes' in target:
                boxes, labels, areas = [], [], []
                for i, mask in enumerate(target['masks']):
                    pos = torch.where(mask)
                    if len(pos[0]) > 0 and len(pos[1]) > 0:
                        y1, y2 = pos[0].min().item(), pos[0].max().item()
                        x1, x2 = pos[1].min().item(), pos[1].max().item()
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(target['labels'][i].item())
                            areas.append((y2 - y1) * (x2 - x1))
                if boxes:
                    target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                    target['labels'] = torch.tensor(labels, dtype=torch.int64)
                    if 'area' in target:
                        target['area'] = torch.tensor(areas, dtype=torch.float32)

        return image, target

def build_mask_rcnn_transforms(train=True, min_size=800, max_size=1333, augmentation_level=2):
    """
    构建Mask R-CNN数据增强转换流水线
    
    Args:
        train: 是否为训练模式
        min_size: 缩放后的最小尺寸
        max_size: 缩放后的最大尺寸
        augmentation_level: 数据增强级别 (1-4)
            - 1级: 最基础的增强，适合实例分割任务的必要增强
            - 2级: 默认级别，中等强度增强，适合大多数实例分割任务
            - 3级: 较强增强，添加更多变换
            - 4级: 最强增强，包含所有可用的增强方法
    
    Returns:
        Compose: 数据增强转换流水线
    """
    transforms = []
    if train:
        # 第1级增强：最基础的增强，适合实例分割任务的必要增强
        if augmentation_level >= 1:
            transforms.extend([
                RandomHorizontalFlip(0.5),  # 水平翻转，基础且有效的增强
            ])
        
        # 第2级增强：默认级别，中等强度增强
        if augmentation_level >= 2:
            transforms.extend([
                SafeRandomCrop(max_crop_fraction=0.2, min_instance_area=0.8, prob=0.3),  # 安全随机裁剪
                SmallRotation(angle_range=10, prob=0.3),  # 小角度旋转，保持实例完整性
                LargeScaleJitter(min_scale=0.3, max_scale=2.0, prob=0.5),  # 尺度抖动，适应不同大小的实例
            ])
        
        # 第3级增强：较强增强
        if augmentation_level >= 3:
            transforms.extend([
                ColorJitterTransform(prob=0.8),  # 颜色抖动，增加颜色多样性
                RandomGrayscale(prob=0.1),  # 随机灰度化
            ])
        
        # 第4级增强：最强增强，包含所有方法
        if augmentation_level >= 4:
            transforms.extend([
                MotionBlur(kernel_size=7, angle_range=180, prob=0.3),  # 运动模糊
                RandomPerspective(distortion_scale=0.2, prob=0.3),  # 随机透视变换
            ])

    # 必要的基础转换，不受augmentation_level影响
    transforms.extend([
        Resize(min_size, max_size),
        Normalize(),
    ])

    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))
