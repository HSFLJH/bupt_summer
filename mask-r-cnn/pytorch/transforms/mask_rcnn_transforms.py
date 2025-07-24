import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter

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

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            _, _, width = image.shape

            if 'boxes' in target:
                boxes = target['boxes']
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
            if 'masks' in target:
                target['masks'] = target['masks'].flip(-1)

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
        
        # 保存原始尺寸
        _, h, w = image.shape
        
        # 将tensor转为PIL进行旋转
        pil_image = F.to_pil_image(image)
        rotated_pil = pil_image.rotate(angle, expand=self.expand, resample=Image.BILINEAR)
        
        # 转回tensor
        rotated_image = F.to_tensor(rotated_pil)
        
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
                    areas = []
                    
                    for i, mask in enumerate(target['masks']):
                        pos = torch.where(mask)
                        if len(pos[0]) > 0 and len(pos[1]) > 0:
                            y1, y2 = pos[0].min().item(), pos[0].max().item()
                            x1, x2 = pos[1].min().item(), pos[1].max().item()
                            
                            # 确保框有效（不是单点）
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(target['labels'][i].item())
                                areas.append((y2 - y1) * (x2 - x1))
                    
                    if boxes:
                        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                        target['labels'] = torch.tensor(labels, dtype=torch.int64)
                        # 更新面积
                        if 'area' in target:
                            target['area'] = torch.tensor(areas, dtype=torch.float32)
        
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
    """随机透视变换，模拟不同视角"""
    def __init__(self, distortion_scale=0.2, prob=0.3):
        """
        参数:
            distortion_scale: 透视变换的强度
            prob: 应用变换的概率
        """
        self.distortion_scale = distortion_scale
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # 将tensor转为PIL进行透视变换
        pil_image = F.to_pil_image(image)
        
        # 获取图像尺寸
        width, height = pil_image.size
        
        # 定义扭曲参数
        half_width = width // 2
        half_height = height // 2
        
        # 计算变换的四个点
        topleft = [
            int(random.uniform(0, self.distortion_scale * half_width)),
            int(random.uniform(0, self.distortion_scale * half_height))
        ]
        topright = [
            int(random.uniform(width - self.distortion_scale * half_width, width)),
            int(random.uniform(0, self.distortion_scale * half_height))
        ]
        botright = [
            int(random.uniform(width - self.distortion_scale * half_width, width)),
            int(random.uniform(height - self.distortion_scale * half_height, height))
        ]
        botleft = [
            int(random.uniform(0, self.distortion_scale * half_width)),
            int(random.uniform(height - self.distortion_scale * half_height, height))
        ]
        
        # 原始图像的四个角点
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [tuple(topleft), tuple(topright), tuple(botright), tuple(botleft)]
        
        # 计算透视变换矩阵
        coeffs = F._get_perspective_coeffs(startpoints, endpoints)
        
        # 应用透视变换
        transformed_image = pil_image.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            Image.BILINEAR
        )
        
        # 转回tensor
        transformed_tensor = F.to_tensor(transformed_image)
        
        # 处理目标数据
        # 对于透视变换，掩码和边界框的处理比较复杂
        # 这里我们简化处理：对掩码应用相同的透视变换，然后重新计算边界框
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks']
            transformed_masks = []
            
            for mask in masks:
                mask_pil = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
                transformed_mask = mask_pil.transform(
                    (width, height),
                    Image.PERSPECTIVE,
                    coeffs,
                    Image.NEAREST
                )
                transformed_mask_np = np.array(transformed_mask) > 0
                transformed_masks.append(torch.tensor(transformed_mask_np, dtype=torch.uint8))
            
            if transformed_masks:
                target['masks'] = torch.stack(transformed_masks)
                
                # 从掩码重新计算边界框
                if 'boxes' in target:
                    boxes = []
                    labels = []
                    areas = []
                    
                    for i, mask in enumerate(target['masks']):
                        pos = torch.where(mask)
                        if len(pos[0]) > 0 and len(pos[1]) > 0:
                            y1, y2 = pos[0].min().item(), pos[0].max().item()
                            x1, x2 = pos[1].min().item(), pos[1].max().item()
                            
                            # 确保框有效（不是单点）
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(target['labels'][i].item())
                                areas.append((y2 - y1) * (x2 - x1))
                    
                    if boxes:
                        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                        target['labels'] = torch.tensor(labels, dtype=torch.int64)
                        # 更新面积
                        if 'area' in target:
                            target['area'] = torch.tensor(areas, dtype=torch.float32)
        
        return transformed_tensor, target

def build_mask_rcnn_transforms(train=True, min_size=800, max_size=1333):
    transforms = []
    if train:
        transforms.extend([
            LargeScaleJitter(min_scale=0.3, max_scale=2.0, prob=0.5),
            RandomHorizontalFlip(0.5),
            ColorJitterTransform(prob=0.8),
            RandomGrayscale(prob=0.1),
            SmallRotation(angle_range=10, prob=0.3),
            SafeRandomCrop(max_crop_fraction=0.2, min_instance_area=0.8, prob=0.3),
            MotionBlur(kernel_size=7, angle_range=180, prob=0.3),
            RandomPerspective(distortion_scale=0.2, prob=0.3),
        ])

    transforms.extend([
        Resize(min_size, max_size),
        Normalize(),
    ])

    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))
