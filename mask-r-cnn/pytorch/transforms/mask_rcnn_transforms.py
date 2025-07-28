import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from typing import Dict, List, Optional, Tuple

# ============================== 数据增强 ==============================

class LargeScaleJitter:
    """
    大尺度抖动（缩放）变换。
    以一定的概率，将图像和其对应的标注（边界框、掩码、面积）按一个随机比例进行缩放。
    这有助于模型学习适应不同尺寸的目标。
    """
    def __init__(self, min_scale: float = 0.1, max_scale: float = 2.0, prob: float = 1.0):
        """
        初始化方法。
        Args:
            min_scale (float): 随机缩放的最小比例。
            max_scale (float): 随机缩放的最大比例。
            prob (float): 执行此变换的概率。
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        # 以 prob 的概率决定是否执行此变换
        if random.random() > self.prob:
            return image, target

        # 获取图像的高和宽
        _, h, w = image.shape
        # 在 [min_scale, max_scale] 范围内随机选择一个缩放比例
        scale = random.uniform(self.min_scale, self.max_scale)
        # 计算缩放后的新尺寸
        nh, nw = int(h * scale), int(w * scale)

        # 使用双线性插值(BILINEAR)对图像进行缩放
        image = F.resize(image, [nh, nw], interpolation=InterpolationMode.BILINEAR)

        # 同步更新标注信息
        if target is not None:
            if 'boxes' in target:
                # 边界框坐标直接乘以缩放比例
                target['boxes'] = target['boxes'] * scale
            if 'area' in target:
                # 面积是二维的，所以乘以缩放比例的平方
                target['area'] = target['area'] * (scale ** 2)
            if 'masks' in target:
                # 对掩码进行缩放，使用最近邻插值(NEAREST)以保持掩码的像素值(0或1)
                masks = target['masks'].unsqueeze(1).float() # 增加一个通道维度以满足interpolate的要求
                masks = torch.nn.functional.interpolate(masks, size=(nh, nw), mode='nearest')
                target['masks'] = masks.squeeze(1).byte() # 移除通道维度并转回byte类型
        
        return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    随机水平翻转。
    继承自 torchvision 的 RandomHorizontalFlip,并重写了 `forward` 方法，
    使其能够同时处理图像和对应的标注信息（边界框和掩码）。
    """
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        执行翻转操作。
        """
        # torch.rand(1) 生成一个0到1的随机数，如果小于设定的概率 p，则执行翻转
        if torch.rand(1) < self.p:
            # 水平翻转图像
            image = F.hflip(image)
            # 如果存在标注信息，也要同步翻转
            if target is not None:
                # 获取图像翻转后的宽度
                _, _, width = F.get_dimensions(image)
                # 翻转边界框的 x 坐标。
                # 原始的 x_min, x_max 变为 width - x_max, width - x_min
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    # 翻转掩码，沿着宽度方向（最后一个维度）进行翻转
                    target["masks"] = target["masks"].flip(-1)
        return image, target

class ColorJitterTransform:
    """
    颜色抖动变换。
    这是一个包装类，以一定的概率对图像的亮度、对比度、饱和度和色调进行随机调整。
    这个变换只影响图像，不影响标注。
    """
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1, prob: float = 0.8):
        """
        初始化方法。
        Args:
            brightness (float): 亮度调整因子。
            contrast (float): 对比度调整因子。
            saturation (float): 饱和度调整因子。
            hue (float): 色调调整因子。
            prob (float): 执行此变换的概率。
        """
        # 创建一个 torchvision 的 ColorJitter 实例
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() < self.prob:
            image = self.jitter(image)
        return image, target

class RandomGrayscale:
    """
    随机灰度化。
    以一定的概率将彩色图像转换为灰度图像。
    这个变换只影响图像，不影响标注。
    """
    def __init__(self, prob: float = 0.1):
        """
        初始化方法。
        Args:
            prob (float): 执行此变换的概率。
        """
        self.prob = prob

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() < self.prob:
            # 创建一个 p=1.0 的 RandomGrayscale 实例并立即调用
            # p=1.0 确保一旦决定要灰度化，就一定会执行
            image = T.RandomGrayscale(p=1.0)(image)
        return image, target

class Normalize:
    """
    标准化变换。
    使用给定的均值和标准差对图像张量进行标准化。
    这是训练深度学习模型前一个非常关键和常规的步骤。
    公式为: `output = (input - mean) / std`
    """
    def __init__(self, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        初始化方法。
        Args:
            mean (Tuple): 图像各通道的均值,通常使用ImageNet数据集的统计值。
            std (Tuple): 图像各通道的标准差,通常使用ImageNet数据集的统计值。
        """
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Resize:
    """
    调整图像尺寸。
    将图像的短边缩放到 `min_size`，同时确保长边不超过 `max_size`，并保持图像的原始长宽比。
    这在目标检测模型(如Faster R-CNN, Mask R-CNN)中是标准的预处理步骤。
    """
    def __init__(self, min_size: int = 800, max_size: int = 1333):
        """
        初始化方法。
        Args:
            min_size (int): 图像短边的目标尺寸。
            max_size (int): 图像长边允许的最大尺寸。
        """
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        _, h, w = image.shape
        # 计算将短边缩放到 min_size 所需的比例
        scale = self.min_size / min(h, w)
        # 检查如果按此比例缩放，长边是否会超过 max_size
        if max(h, w) * scale > self.max_size:
            # 如果超过了，则以将长边缩放到 max_size 为准，重新计算比例
            scale = self.max_size / max(h, w)

        # 计算缩放后的新尺寸
        nh, nw = int(h * scale), int(w * scale)
        # 使用双线性插值调整图像尺寸
        image = F.resize(image, [nh, nw], interpolation=InterpolationMode.BILINEAR)

        # 同步更新标注信息，逻辑同 LargeScaleJitter
        if target is not None:
            if 'boxes' in target:
                target['boxes'] = target['boxes'] * scale
            if 'area' in target:
                target['area'] = target['area'] * (scale ** 2)
            if 'masks' in target:
                masks = target['masks'].unsqueeze(1).float()
                masks = torch.nn.functional.interpolate(masks, size=(nh, nw), mode='nearest')
                target['masks'] = masks.squeeze(1).byte()
        
        return image, target

class SmallRotation:
    """
    小角度旋转变换。
    以一定概率对图像和掩码进行小角度的随机旋转。
    旋转后，会根据旋转后的掩码重新计算边界框，以确保边界框紧密地包围实例。
    """
    def __init__(self, angle_range: int = 10, prob: float = 0.3, expand: bool = False):
        """
        初始化方法。
        Args:
            angle_range (int): 旋转角度的范围，将在 [-angle_range, +angle_range] 之间随机选择。
            prob (float): 执行此变换的概率。
            expand (bool): 如果为 True，则扩展图像尺寸以包含整个旋转后的图像。
        """
        self.angle_range = angle_range
        self.prob = prob
        self.expand = expand
        # 为图像和掩码设置不同的插值方法
        self.interpolation_image = F.InterpolationMode.BILINEAR # 图像用双线性插值，效果平滑
        self.interpolation_mask = F.InterpolationMode.NEAREST   # 掩码用最近邻插值，保持像素值
    
    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() > self.prob:
            return image, target
        
        # 随机选择旋转角度
        angle = random.uniform(-self.angle_range, self.angle_range)

        # 旋转图像
        image = F.rotate(image, angle, interpolation=self.interpolation_image, expand=self.expand)

        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks']
            rotated_masks = []
            # 逐个旋转每个实例的掩码
            for mask in masks:
                # 增加一个伪批次/通道维度以满足 rotate 函数的输入要求
                rotated = F.rotate(mask.unsqueeze(0), angle, interpolation=self.interpolation_mask, expand=self.expand)
                rotated_masks.append(rotated.squeeze(0))
            target['masks'] = torch.stack(rotated_masks) # 将旋转后的掩码列表堆叠成一个张量

            # 关键步骤：根据旋转后的掩码重新计算边界框、标签和面积
            if 'boxes' in target:
                boxes, labels, areas = [], [], []
                # 遍历每个新的掩码来找到其新的边界框
                for i, mask in enumerate(target['masks']):
                    pos = torch.where(mask) # 找到掩码中所有非零像素的位置
                    # 确保掩码不为空
                    if len(pos[0]) > 0 and len(pos[1]) > 0:
                        # 找到 y 和 x 坐标的最小值和最大值，形成新的边界框
                        y1, y2 = pos[0].min().item(), pos[0].max().item()
                        x1, x2 = pos[1].min().item(), pos[1].max().item()
                        # 确保是一个有效的框 (width > 0 and height > 0)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(target['labels'][i].item())
                            areas.append((y2 - y1) * (x2 - x1))
                
                # 如果有有效的框保留下来，则更新 target
                if boxes:
                    target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                    target['labels'] = torch.tensor(labels, dtype=torch.int64)
                    if 'area' in target:
                        target['area'] = torch.tensor(areas, dtype=torch.float32)
                else: # 如果所有实例都被转出边界或变得无效，则清空 target
                    target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                    target['labels'] = torch.empty((0,), dtype=torch.int64)
                    if 'masks' in target: target['masks'] = torch.empty((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
                    if 'area' in target: target['area'] = torch.empty((0,), dtype=torch.float32)

        return image, target

class SafeRandomCrop:
    """
    安全随机裁剪。
    以一定概率从图像的边缘随机裁剪掉一部分，但会尽量保证裁剪后的图像中，
    原有的物体实例（instance）大部分被保留下来。
    """
    def __init__(self, max_crop_fraction: float = 0.2, min_instance_area: float = 0.8, prob: float = 0.3):
        """
        初始化方法。
        Args:
            max_crop_fraction (float): 相对于原始尺寸，从每个边缘裁剪的最大比例。
            min_instance_area (float): 一个实例被保留下来所需的最小面积比例。
                                       例如 0.8 表示实例的80%以上必须在裁剪框内。
            prob (float): 执行此变换的概率。
        """
        self.max_crop_fraction = max_crop_fraction
        self.min_instance_area = min_instance_area
        self.prob = prob
    
    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() > self.prob:
            return image, target
        
        _, h, w = image.shape
        
        # 随机决定从四个方向（上、下、左、右）各裁剪多少
        max_crop_x = int(w * self.max_crop_fraction)
        max_crop_y = int(h * self.max_crop_fraction)
        crop_left = random.randint(0, max_crop_x)
        crop_top = random.randint(0, max_crop_y)
        crop_right = random.randint(0, max_crop_x)
        crop_bottom = random.randint(0, max_crop_y)
        
        # 计算裁剪后的新图像区域坐标
        new_left = crop_left
        new_top = crop_top
        new_right = w - crop_right
        new_bottom = h - crop_bottom
        
        # 如果裁剪区域无效（例如左边比右边还大），则不进行裁剪
        if new_left >= new_right or new_top >= new_bottom:
            return image, target
        
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            keep_indices = [] # 用于存储被保留下来的实例的索引
            new_boxes = [] # 用于存储裁剪后更新的边界框
            
            for i, box in enumerate(boxes):
                # 计算原始边界框和裁剪区域的交集
                ix1 = max(box[0], new_left)
                iy1 = max(box[1], new_top)
                ix2 = min(box[2], new_right)
                iy2 = min(box[3], new_bottom)
                
                # 如果交集有效
                if ix2 > ix1 and iy2 > iy1:
                    # 计算交集面积占原始边界框面积的比例
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    overlap_area = (ix2 - ix1) * (iy2 - iy1)
                    # 避免除以零
                    if box_area > 0:
                        ratio = overlap_area / box_area
                    else:
                        ratio = 0
                    
                    # 如果比例大于等于设定的阈值，则保留这个实例
                    if ratio >= self.min_instance_area:
                        keep_indices.append(i)
                        # 将边界框坐标更新为相对于裁剪后图像的新坐标
                        new_boxes.append([
                            ix1 - new_left,
                            iy1 - new_top,
                            ix2 - new_left,
                            iy2 - new_top
                        ])
            
            # 如果裁剪后没有任何实例被保留下来，则放弃此次裁剪
            if not keep_indices:
                return image, target
            
            # 执行裁剪
            cropped_image = image[:, new_top:new_bottom, new_left:new_right]
            
            # 更新标注信息
            target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
            target['labels'] = target['labels'][keep_indices]
            
            if 'masks' in target:
                masks = target['masks'][keep_indices] # 先只选择被保留的掩码
                # 然后对这些掩码进行裁剪
                target['masks'] = masks[:, new_top:new_bottom, new_left:new_right]
            
            if 'area' in target:
                # 根据新的边界框重新计算面积
                target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
            
            return cropped_image, target
        
        # 如果图像中本来就没有标注框，直接进行裁剪
        cropped_image = image[:, new_top:new_bottom, new_left:new_right]
        return cropped_image, target

class MotionBlur:
    """
    运动模糊变换。
    以一定概率为图像添加运动模糊效果，模拟相机或物体在曝光期间的运动。
    """
    def __init__(self, kernel_size: int = 7, angle_range: int = 180, prob: float = 0.3):
        """
        初始化方法。
        Args:
            kernel_size (int): 模糊核的大小，必须是奇数。
            angle_range (int): 运动方向的角度范围 (0-180度)。
            prob (float): 执行此变换的概率。
        """
        self.kernel_size = kernel_size
        self.angle_range = angle_range
        self.prob = prob
    
    def _motion_blur_kernel(self, kernel_size: int, angle: float) -> np.ndarray:
        """
        内部辅助函数，用于创建一个运动模糊的卷积核。
        """
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        
        # 在核的中心画一条线来模拟运动轨迹
        for i in range(kernel_size):
            offset = i - center
            x = center + dx * offset
            y = center + dy * offset
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                # 使用双线性插值将线上的能量分布到周围的整数像素点
                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, kernel_size - 1), min(y0 + 1, kernel_size - 1)
                wx, wy = x - x0, y - y0
                kernel[y0, x0] += (1 - wx) * (1 - wy)
                kernel[y0, x1] += wx * (1 - wy)
                kernel[y1, x0] += (1 - wx) * wy
                kernel[y1, x1] += wx * wy
        
        # 归一化，使核内所有值的和为1
        return kernel / kernel.sum()
    
    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() > self.prob:
            return image, target
        
        angle = random.uniform(0, self.angle_range)
        
        # 创建模糊核并转换为 PyTorch 张量
        kernel = self._motion_blur_kernel(self.kernel_size, angle)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0) # 形状变为 [1, 1, k, k]
        
        # 应用模糊
        # 将核复制3份，分别对应 R, G, B 三个通道
        kernel = kernel.repeat(3, 1, 1, 1)
        
        # 使用分组卷积（groups=3）对每个通道独立应用模糊核
        # 这是一种高效实现对多通道图像应用相同2D滤波的方法
        padding = self.kernel_size // 2
        blurred_image = torch.nn.functional.conv2d(
            image.unsqueeze(0), # 增加一个批次维度 [1, C, H, W]
            kernel, 
            padding=padding, 
            groups=3 # C_in=3, C_out=3, groups=3, 表示深度可分离卷积
        ).squeeze(0) # 移除批次维度
        
        # 确保图像像素值仍在有效范围内 [0, 1]
        blurred_image = torch.clamp(blurred_image, 0, 1)
        
        return blurred_image, target

class RandomPerspective:
    """
    随机透视变换。
    以一定概率对图像和掩码进行随机的透视变换。
    类似于 `SmallRotation`，变换后也会根据新的掩码重新计算边界框。
    """
    def __init__(self, distortion_scale: float = 0.2, prob: float = 0.3):
        """
        初始化方法。
        Args:
            distortion_scale (float): 扭曲程度的缩放因子。值越大，扭曲越明显。
            prob (float): 执行此变换的概率。
        """
        self.distortion_scale = distortion_scale
        self.prob = prob
        self.interpolation_image = F.InterpolationMode.BILINEAR
        self.interpolation_mask = F.InterpolationMode.NEAREST

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() > self.prob:
            return image, target
        
        _, h, w = image.shape

        # 定义原图的四个角点作为起始点
        startpoints = [
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ]

        # 定义目标点：在原图角点的基础上，添加随机扰动
        def distort(pt):
            dx = random.uniform(-self.distortion_scale, self.distortion_scale) * w
            dy = random.uniform(-self.distortion_scale, self.distortion_scale) * h
            return [pt[0] + dx, pt[1] + dy]
        endpoints = [distort(pt) for pt in startpoints]

        # 对图像进行透视变换
        image = F.perspective(image, startpoints, endpoints, interpolation=self.interpolation_image)

        # 对掩码进行透视变换，并重新计算标注
        if 'masks' in target and len(target['masks']) > 0:
            new_masks = []
            for mask in target['masks']:
                mask_transformed = F.perspective(mask.unsqueeze(0), startpoints, endpoints, interpolation=self.interpolation_mask)
                new_masks.append(mask_transformed.squeeze(0))
            target['masks'] = torch.stack(new_masks)

            # 同样地，根据变换后的掩码重新计算边界框等信息
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
                else: # 清空 target
                    target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                    target['labels'] = torch.empty((0,), dtype=torch.int64)
                    if 'masks' in target: target['masks'] = torch.empty((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
                    if 'area' in target: target['area'] = torch.empty((0,), dtype=torch.float32)

        return image, target

# ============================== 组合函数 ==============================

class Compose:
    """
    一个变换的容器，可以将多个变换操作组合在一起，并按顺序依次执行。
    这在数据预处理流程中非常常见，可以将一系列增强步骤串联起来。
    """
    def __init__(self, transforms: List[object]):
        """
        初始化方法。
        Args:
            transforms (List[object]): 一个包含多个变换操作实例的列表。
        """
        self.transforms = transforms

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        使得类的实例可以像函数一样被调用。
        它会遍历列表中的每一个变换 `t`，并将图像 `image` 和标注 `target` 传入，
        然后用返回的新 `image` 和 `target` 更新原来的变量，实现链式调用。
        
        Args:
            image (Tensor): 输入的图像张量。
            target (Dict): 输入的标注信息，通常是一个字典，包含 "boxes", "masks", "labels" 等。
            
        Returns:
            Tuple[Tensor, Dict]: 经过所有变换处理后的图像和标注。
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# ============================== 构建数据增强流水线 ==============================

def build_mask_rcnn_transforms(train: bool = True, min_size: int = 800, max_size: int = 1333, augmentation_level: int = 2):
    """
    构建用于 Mask R-CNN 模型的数据增强和预处理流水线。
    
    这个函数非常灵活，可以根据是否是训练阶段以及所需的数据增强强度，来组合不同的变换。
    
    Args:
        train (bool): 如果为 True,则构建包含数据增强的训练流水线。
                      如果为 False,则只构建包含必要预处理(如Resize和Normalize)的验证/测试流水线。
        min_size (int): 调整尺寸后的图像短边最小尺寸。
        max_size (int): 调整尺寸后的图像长边最大尺寸。
        augmentation_level (int): 数据增强的级别 (1-4)。级别越高，增强越复杂、越强烈。
            - 1级: 基础增强，只有水平翻转。
            - 2级: 中等增强,在1级基础上增加安全裁剪、小角度旋转和尺度抖动。这是默认级别。
            - 3级: 较强增强,在2级基础上增加颜色抖动和随机灰度化。
            - 4级: 最强增强,在3级基础上增加运动模糊和透视变换。
    
    Returns:
        Compose: 一个包含了所选变换的 `Compose` 对象。
    """
    transforms = []
    if train:
        # --- 根据增强级别添加不同的变换 ---
        
        # 级别 >= 1: 添加最基础、最常用且最有效的增强
        if augmentation_level >= 1:
            transforms.extend([
                RandomHorizontalFlip(0.5), # 50%的概率水平翻转
            ])
        
        # 级别 >= 2: 添加几何变换，有助于模型应对物体的位置、旋转和大小变化
        if augmentation_level >= 2:
            transforms.extend([
                SafeRandomCrop(prob=0.3), # 安全随机裁剪
                SmallRotation(prob=0.3),  # 小角度旋转
                LargeScaleJitter(prob=0.5), # 尺度抖动
            ])
        
        # 级别 >= 3: 添加颜色相关的变换，增强模型对光照、色彩变化的鲁棒性
        if augmentation_level >= 3:
            transforms.extend([
                ColorJitterTransform(prob=0.8), # 颜色抖动
                RandomGrayscale(prob=0.1),      # 随机灰度化
            ])
        
        # 级别 >= 4: 添加更复杂、模拟真实世界场景的变换
        if augmentation_level >= 4:
            transforms.extend([
                MotionBlur(prob=0.3),         # 运动模糊
                RandomPerspective(prob=0.3),  # 随机透视变换
            ])

    # --- 添加所有模式下都必需的基础变换 ---
    # 无论是否训练，都需要将图像调整到合适的尺寸并进行标准化
    transforms.extend([
        # 这里没有 ToTensor()，因为我们假设输入已经是 Tensor 了。
        # 如果输入是 PIL.Image，需要在这里加上一个 ToTensor()。
        Resize(min_size, max_size),
        Normalize(),
    ])

    return Compose(transforms)

# ============================== 辅助函数： ==============================

def collate_fn(batch: List[Tuple[Tensor, Dict]]) -> Tuple[List, List]:
    """
    一个整理函数（collate function），用于 PyTorch 的 DataLoader。
    在目标检测或实例分割任务中，一个批次（batch）中的每张图片尺寸可能不同，
    每个`target`字典中的目标数量也不同，因此不能直接用 `torch.stack` 合并成一个大张量。
    
    这个函数的作用是：将一个批次的 `[(image1, target1), (image2, target2), ...]`
    转换成 `([image1, image2, ...], [target1, target2, ...])` 的形式。
    这样模型就可以方便地处理一个图像列表和一个标注列表。
    
    Args:
        batch (List): 一个列表，每个元素都是一个 (image, target) 元组。
        
    Returns:
        Tuple: 一个元组，第一个元素是图像列表，第二个元素是标注字典列表。
    """
    return tuple(zip(*batch))
