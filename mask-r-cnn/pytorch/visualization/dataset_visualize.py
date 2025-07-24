"""
Mask R-CNN的可视化模块
包含用于可视化图像、边界框和掩码的函数
"""
import os
import numpy as np
import matplotlib 
# 设置后端为Agg（无界面后端）
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
import cv2
from PIL import Image, ImageDraw
import random
import torchvision.transforms.functional as F
from transforms.mask_rcnn_transforms import (
    LargeScaleJitter, RandomHorizontalFlip, ColorJitterTransform,
    RandomGrayscale, SmallRotation, SafeRandomCrop, Normalize,
    ToTensor, Resize, MotionBlur, RandomPerspective
)


import io
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import webbrowser
import socket
import time

# 全局变量存储图像缓存
image_cache = {}
server_thread = None
server_instance = None
server_port = None

def tensor_to_numpy(image):
    """
    将tensor图像转换为numpy数组，用于matplotlib显示
    
    Args:
        image: torch.Tensor类型的图像，形状为(C,H,W)
    
    Returns:
        numpy数组，形状为(H,W,C)，值域为[0,255]的uint8类型
    """
    if not isinstance(image, torch.Tensor):
        return np.array(image)
        
    # 反标准化，将数据范围恢复到[0, 1]
    if image.shape[0] == 3:  # 如果是标准化的张量(C, H, W)
        # 假设使用的是ImageNet的均值和标准差
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
    
    # 转换为numpy数组
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    
    return image_np

def visualize_sample_with_annotations(image, target=None, title=None, figsize=(12, 12), show_annotations=True):
    """
    可视化一个图像样本及其实例分割标注
    
    Args:
        image: PIL.Image或torch.Tensor类型的图像
        target: 包含'boxes'和'masks'的字典，如果为None则只显示图像
        title: 图像标题
        figsize: 图像大小
        show_annotations: 是否显示标注（边界框和掩码）
    """
    # 将图像转换为numpy数组
    image_np = tensor_to_numpy(image)
    
    # 创建matplotlib图像
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_np)
    
    # 如果需要显示标注且有标注数据
    if show_annotations and target is not None:
        # 绘制每个实例的掩码和边界框
        colors = generate_colors(len(target.get('masks', [])))
        
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks']
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            
            for i, mask in enumerate(masks):
                color = colors[i]
                
                # 绘制掩码
                mask_image = np.zeros_like(image_np)
                if len(mask.shape) == 2:  # 单通道掩码
                    mask_bool = mask > 0
                    for c in range(3):
                        mask_image[:, :, c] = np.where(mask_bool, color[c], 0)
                
                # 透明掩码叠加
                mask_image = cv2.addWeighted(image_np, 1, mask_image, 0.5, 0)
                ax.imshow(mask_image, alpha=0.5)
        
        # 绘制边界框
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                color = colors[i]
                
                # matplotlib使用RGB格式，转换为0-1范围
                rgb_color = [c/255 for c in color]
                
                # 绘制边界框
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    fill=False, edgecolor=rgb_color, linewidth=2)
                ax.add_patch(rect)
                
                # 绘制标签
                if 'labels' in target:
                    label_id = target['labels'][i]
                    if isinstance(label_id, torch.Tensor):
                        label_id = label_id.item()
                    ax.text(x1, y1, f'ID:{label_id}', 
                            bbox=dict(facecolor=rgb_color, alpha=0.5),
                            fontsize=10, color='white')
    
    if title:
        ax.set_title(title)
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def generate_colors(n):
    """
    生成n个不同的颜色
    
    Args:
        n: 颜色数量
    
    Returns:
        colors: 颜色列表，每个颜色为RGB元组
    """
    colors = []
    for i in range(n):
        # 使用HSV色彩空间确保颜色多样性
        h = i / n
        s = 0.8 + random.random() * 0.2  # 0.8-1.0
        v = 0.8 + random.random() * 0.2  # 0.8-1.0
        
        # 转换为RGB
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            # 创建HTML页面，列出所有图像
            html = '<html><head><meta charset="UTF-8"><title>Mask R-CNN 数据增强可视化</title>'
            html += '<style>body {font-family: Arial, "Microsoft YaHei", sans-serif; margin: 20px;} '
            html += 'h1 {color: #333;} '
            html += '.image-container {margin-bottom: 40px; border: 1px solid #ddd; padding: 15px; border-radius: 5px;} '
            html += 'img {max-width: 100%; height: auto; border: 1px solid #eee;} '
            html += '</style></head><body>'
            html += '<h1>Mask R-CNN 数据增强可视化</h1>'
            
            for key in image_cache:
                html += f'<div class="image-container"><h2>{key}</h2>'
                html += f'<a href="/image/{key}" target="_blank"><img src="data:image/png;base64,{image_cache[key]}" /></a>'
                html += '</div>'
            
            html += '</body></html>'
            self.wfile.write(html.encode('utf-8'))
        
        elif self.path.startswith('/image/'):
            image_key = self.path[7:]
            if image_key in image_cache:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(base64.b64decode(image_cache[image_key]))
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Image not found')
    
    def log_message(self, format, *args):
        # 禁用请求日志
        return

def find_free_port():
    """找到一个可用的端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_server():
    global server_instance, server_port
    
    # 找到可用端口
    port = find_free_port()
    server_port = port
    
    server_instance = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)
    print(f"\n=== 启动可视化服务器于端口 {port} ===")
    print(f"请在VSCode中设置端口转发 {port} -> {port}")
    print(f"然后在浏览器中访问 http://localhost:{port}/")
    print("如果您使用的是VSCode Remote，端口转发可能已自动设置")
    print("==================================\n")
    
    # 启动服务器
    server_instance.serve_forever()

def stop_server():
    """停止HTTP服务器"""
    global server_instance, server_thread
    
    if server_instance:
        server_instance.shutdown()
        server_instance = None
    
    if server_thread:
        server_thread.join()
        server_thread = None

def save_figure_to_cache(fig, key):
    """将matplotlib图形保存到图像缓存"""
    global image_cache
    
    # 将图像保存到内存
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # 转换为base64编码并存储
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    image_cache[key] = img_str

def preview_augmentations(dataset, num_samples=5, config=None, sample_indices=None, show_annotations=True):
    """
    预览各种数据增强效果
    
    Args:
        dataset: 数据集对象
        num_samples: 要显示的样本数量
        config: 配置对象，包含变换参数
        sample_indices: 指定的样本索引列表，如果提供则优先使用
        show_annotations: 是否显示标注（边界框和掩码）
    """
    global server_thread, server_instance
    
    if num_samples <= 0:
        return
    
    # 启动HTTP服务器（如果尚未启动）
    if server_thread is None:
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        # 等待服务器启动
        time.sleep(1)
    
    # 清空图像缓存
    image_cache.clear()
    
    # 选择样本
    if sample_indices is not None:
        # 使用指定的样本索引
        indices = [idx for idx in sample_indices if 0 <= idx < len(dataset)]
        if not indices:  # 如果所有指定的索引都无效
            print("指定的样本索引无效，将使用随机样本")
            # 重新设置随机种子，使其基于当前时间
            random.seed(int(time.time()))
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    else:
        # 重新设置随机种子，使其基于当前时间
        random.seed(int(time.time()))
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    print(f"\n选择的样本索引: {indices}")
    
    # 设置augmentations,即数据增强方式以及参数
    augmentations = {
        "Original": None,
        "Large Scale Jitter": LargeScaleJitter(min_scale=0.3, max_scale=2.0),
        "Random Horizontal Flip": RandomHorizontalFlip(prob=1.0),  # 设置为1.0确保应用
        "Color Jitter": ColorJitterTransform(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=1.0),
        "Grayscale": RandomGrayscale(prob=1.0),
        "Small Rotation": SmallRotation(angle_range=10, prob=1.0),
        "Safe Random Crop": SafeRandomCrop(max_crop_fraction=0.2, min_instance_area=0.8, prob=1.0),
        "Motion Blur": MotionBlur(kernel_size=7, angle_range=180, prob=1.0),
        "Random Perspective": RandomPerspective(distortion_scale=0.2, prob=1.0)
    }
    
    # 为每个样本和每种增强创建子图
    for idx in indices:
        img, target = dataset[idx]
        
        # 确保图像是tensor格式
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        
        # 创建原始图像副本供变换使用
        original_img = img.clone()
        original_target = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                           for k, v in target.items()}
        
        # 为每种增强创建一个图
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"Sample {idx} Augmentation Preview", fontsize=16)
        
        # 计算子图布局
        num_augmentations = len(augmentations)
        rows = (num_augmentations + 2) // 3  # 每行最多3个子图
        cols = min(3, num_augmentations)
        
        for i, (aug_name, transform) in enumerate(augmentations.items()):
            # 创建子图
            ax = fig.add_subplot(rows, cols, i+1)
            
            if transform is None:
                # 显示原始图像
                augmented_img = original_img
                augmented_target = original_target
            else:
                # 应用变换并显示
                augmented_img = original_img.clone()
                augmented_target = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                   for k, v in original_target.items()}
                augmented_img, augmented_target = transform(augmented_img, augmented_target)
            
            # 将图像转换为numpy数组
            augmented_img_np = tensor_to_numpy(augmented_img)
            
            # 显示图像
            ax.imshow(augmented_img_np)
            
            # 如果需要显示标注
            if show_annotations:
                # 绘制边界框和掩码
                colors = generate_colors(len(augmented_target.get('masks', [])))
                
                if 'masks' in augmented_target and len(augmented_target['masks']) > 0:
                    masks = augmented_target['masks']
                    if isinstance(masks, torch.Tensor):
                        masks = masks.cpu().numpy()
                    
                    for j, mask in enumerate(masks):
                        color = colors[j]
                        
                        # 绘制掩码
                        mask_image = np.zeros_like(augmented_img_np)
                        if len(mask.shape) == 2:  # 单通道掩码
                            mask_bool = mask > 0
                            for c in range(3):
                                mask_image[:, :, c] = np.where(mask_bool, color[c], 0)
                        
                        # 透明掩码叠加
                        mask_image = cv2.addWeighted(augmented_img_np, 1, mask_image, 0.5, 0)
                        ax.imshow(mask_image, alpha=0.5)
                
                # 绘制边界框
                if 'boxes' in augmented_target and len(augmented_target['boxes']) > 0:
                    boxes = augmented_target['boxes']
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    
                    for j, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        color = colors[j]
                        
                        # matplotlib使用RGB格式，转换为0-1范围
                        rgb_color = [c/255 for c in color]
                        
                        # 绘制边界框
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            fill=False, edgecolor=rgb_color, linewidth=2)
                        ax.add_patch(rect)
            
            ax.set_title(aug_name)
            ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # 为总标题留出空间
        
        # 保存图像到缓存而不是直接显示
        save_figure_to_cache(fig, f"sample_{idx}")
        plt.close(fig)
        
        # 为每种增强创建单独的图像
        for aug_name, transform in augmentations.items():
            augmented_img = original_img.clone()
            augmented_target = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                               for k, v in original_target.items()}
            
            if transform is not None:
                augmented_img, augmented_target = transform(augmented_img, augmented_target)
            
            # 创建单独的图像
            fig = visualize_sample_with_annotations(
                augmented_img, 
                augmented_target,
                title=f"Sample {idx} - {aug_name}",
                show_annotations=show_annotations
            )
            
            # 保存到缓存
            save_figure_to_cache(fig, f"sample_{idx}_{aug_name}")
            plt.close(fig)
    
    print(f"\n可视化已准备完毕! 请访问 http://localhost:{server_port}/ 查看结果")
    print("===== 按Ctrl+C可以停止服务器和程序 =====\n") 

    # 睡眠300秒，然后停止服务器
    time.sleep(300)
    stop_server()

