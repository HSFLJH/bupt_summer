# 完整脚本：Mask R-CNN 推理 + 可视化 + 批量输出 + HTML 服务

import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
import datetime
import numpy as np
import matplotlib.patches as patches
from pycocotools.coco import COCO
import matplotlib

# 使用英文标签避免字体问题
matplotlib.rcParams['axes.unicode_minus'] = False

from dataset_coco import CocoInstanceDataset, get_transform
from model import get_instance_segmentation_model

# ========= 参数设置 =========
data_root = "/home/lishengjie/data/COCO2017/val2017"
ann_file = "/home/lishengjie/data/COCO2017/annotations/instances_val2017.json"
model_path = "/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/torchvision/result/three/model_final.pth"
num_classes = 90 + 1  # COCO class + background

# 创建日期格式的输出目录
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_output_dir = "/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/torchvision/result/result_pngs/seq"
date_dir = os.path.join(base_output_dir, current_date)
os.makedirs(date_dir, exist_ok=True)

# ========= 加载类别名称 =========
def load_category_names(ann_file):
    """
    从 COCO 标注文件中读取类别名称，返回字典和列表
    """
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {cat['id']: cat['name'] for cat in cats}
    id_list = [id_to_name[i] for i in sorted(id_to_name)]
    return id_to_name, id_list

id_to_name, _ = load_category_names(ann_file)

# ========= 模型加载 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_instance_segmentation_model(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# ========= 加载数据 =========
dataset = CocoInstanceDataset(
    img_folder=data_root,
    ann_file=ann_file,
    transforms=get_transform(train=False)
)

# ========= 可视化函数 =========
def visualize_prediction(img, pred, id_to_name, sequence_dir, idx, score_threshold=0.5):
    masks = pred['masks'][:, 0] > 0.5
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']

    # 过滤低置信度预测
    keep = scores > score_threshold
    masks = masks[keep]
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    img_uint8 = (img * 255).byte()
    subimgs = []

    # 1. 原图
    orig_img = to_pil_image(img_uint8)
    subimgs.append(orig_img)
    orig_img.save(os.path.join(sequence_dir, f"2_原图_{idx}.png"))

    # 2. 检测框 + 类别 + 置信度
    img2 = img_uint8.clone()
    img2_plt = plt.figure(figsize=(10, 10))
    ax2 = img2_plt.add_subplot(111)
    ax2.imshow(to_pil_image(img2))
    
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        label_id = labels[i].item()
        label_name = id_to_name.get(label_id, str(label_id))
        score = scores[i].item()
        
        ax2.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='green', facecolor='none'
        ))
        # 添加类别和置信度
        ax2.text(box[0], box[1] - 5, f'{label_name}: {score:.2f}', fontsize=10,
                bbox=dict(facecolor='green', alpha=0.8), color='white')
    
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sequence_dir, f"3_检测框和类别置信度_{idx}.png"), bbox_inches='tight')
    plt.close()
    subimgs.append(Image.open(os.path.join(sequence_dir, f"3_检测框和类别置信度_{idx}.png")))

    # 3. 只有mask
    num_instances = len(masks)
    if num_instances > 0:
        mask_img_plt = plt.figure(figsize=(10, 10))
        ax3 = mask_img_plt.add_subplot(111)
        
        masks_overlay = np.zeros_like(img.permute(1, 2, 0).cpu().numpy())
        colors = plt.get_cmap('hsv', num_instances)
        
        for i in range(num_instances):
            color = colors(i)[:3]  # 只取RGB部分
            mask = masks[i].cpu().numpy()
            masks_overlay[mask] = color
        
        ax3.imshow(masks_overlay, alpha=0.4)

        ax3.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sequence_dir, f"5_只有mask_{idx}.png"), bbox_inches='tight')
        plt.close()
        subimgs.append(Image.open(os.path.join(sequence_dir, f"5_只有mask_{idx}.png")))
    else:
        # 如果没有mask，保存一个空白图像
        blank = Image.new('RGB', (img_uint8.shape[1], img_uint8.shape[2]), color='white')
        blank.save(os.path.join(sequence_dir, f"5_只有mask_{idx}.png"))
        subimgs.append(blank)

    # 4. mask + 原图
    num_instances = len(masks)
    if num_instances > 0:
        mask_img_plt = plt.figure(figsize=(10, 10))
        ax4 = mask_img_plt.add_subplot(111)
        ax4.imshow(to_pil_image(img_uint8))
        
        masks_overlay = np.zeros_like(img.permute(1, 2, 0).cpu().numpy())
        colors = plt.get_cmap('hsv', num_instances)
        
        for i in range(num_instances):
            color = colors(i)[:3]  # 只取RGB部分
            mask = masks[i].cpu().numpy()
            masks_overlay[mask] = color
        
        ax4.imshow(masks_overlay, alpha=0.4)
        ax4.imshow(to_pil_image(img_uint8), alpha=0.5)
        ax4.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sequence_dir, f"5_mask + 原图_{idx}.png"), bbox_inches='tight')
        plt.close()
        subimgs.append(Image.open(os.path.join(sequence_dir, f"5_mask + 原图_{idx}.png")))
    else:
        # 如果没有mask，保存一个空白图像
        blank = Image.new('RGB', (img_uint8.shape[1], img_uint8.shape[2]), color='white')
        blank.save(os.path.join(sequence_dir, f"5_mask + 原图_{idx}.png"))
        subimgs.append(blank)

    # 5. mask + 类别
    if num_instances > 0:
        mask_label_plt = plt.figure(figsize=(10, 10))
        ax5 = mask_label_plt.add_subplot(111)
        ax5.imshow(to_pil_image(img_uint8))
        
        for i in range(num_instances):
            color = colors(i)[:3]
            mask = masks[i].cpu().numpy()
            label_id = labels[i].item()
            label_name = id_to_name.get(label_id, str(label_id))
            
            # 显示mask
            mask_overlay = np.zeros_like(img.permute(1, 2, 0).cpu().numpy())
            mask_overlay[mask] = color
            ax5.imshow(mask_overlay, alpha=0.5)
            
            # 找到mask的中心点来放置标签
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                ax5.text(center_x, center_y, label_name, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7), color='black')
                
        # 5. 最后画上原图
        ax5.imshow(to_pil_image(img_uint8), alpha=0.5)
        
        ax5.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sequence_dir, f"6_mask和类别_{idx}.png"), bbox_inches='tight')
        plt.close()
        subimgs.append(Image.open(os.path.join(sequence_dir, f"6_mask和类别_{idx}.png")))
    else:
        blank = Image.new('RGB', (img_uint8.shape[1], img_uint8.shape[2]), color='white')
        blank.save(os.path.join(sequence_dir, f"6_mask和类别_{idx}.png"))
        subimgs.append(blank)

    # 6. mask + 框 + 类别
    if num_instances > 0:
        mix_plt = plt.figure(figsize=(10, 10))
        ax6 = mix_plt.add_subplot(111)
        ax6.imshow(to_pil_image(img_uint8))
        
        for i in range(num_instances):
            color = colors(i)[:3]
            mask = masks[i].cpu().numpy()
            box = boxes[i].cpu().numpy()
            label_id = labels[i].item()
            label_name = id_to_name.get(label_id, str(label_id))
            
            # 显示mask
            mask_overlay = np.zeros_like(img.permute(1, 2, 0).cpu().numpy())
            mask_overlay[mask] = color
            ax6.imshow(mask_overlay, alpha=0.5)
            
            # 显示边框
            ax6.add_patch(patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor=color, facecolor='none'
            ))
            
            # 显示类别
            ax6.text(box[0], box[1] - 5, label_name, fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.8), color='white')
        
        ax6.imshow(to_pil_image(img_uint8), alpha=0.5)
        ax6.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sequence_dir, f"7_mask框类别_{idx}.png"), bbox_inches='tight')
        plt.close()
        subimgs.append(Image.open(os.path.join(sequence_dir, f"7_mask框类别_{idx}.png")))
    else:
        blank = Image.new('RGB', (img_uint8.shape[1], img_uint8.shape[2]), color='white')
        blank.save(os.path.join(sequence_dir, f"7_mask框类别_{idx}.png"))
        subimgs.append(blank)

    # 创建六张子图拼接
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.flatten()
    
    for i in range(6):
        axs[i].imshow(subimgs[i])
        axs[i].axis('off')
        # 使用英文标题避免字体问题
        axs[i].set_title(["Original", "Boxes & Labels", "Masks Only", 
                           "Masks & Image", "Masks & Labels", "All Features"][i], fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(sequence_dir, f"1_六图拼接_{idx}.png"))
    plt.close()

    return subimgs

# ========= 批量处理 =========
def run_inference(sequence_ids):
    results = []
    sequence_dirs = []
    
    # 为每个序列创建单独的目录
    for idx in sequence_ids:
        # 为这个序列创建一个子目录
        sequence_dir = os.path.join(date_dir, f"序列_{idx}")
        os.makedirs(sequence_dir, exist_ok=True)
        sequence_dirs.append(sequence_dir)
        
        img, _ = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]
        
        subfigs = visualize_prediction(img.cpu(), pred, id_to_name, sequence_dir, idx)
        results.append((idx, sequence_dir))
    
    return results, sequence_dirs

# ========= HTML服务 =========
def generate_html(results, sequence_dirs):
    # 为每个序列生成单独的HTML
    for idx, sequence_dir in results:
        html_path = os.path.join(sequence_dir, "index.html")
        with open(html_path, "w", encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n<head>\n")
            f.write('<meta charset="UTF-8">\n')
            f.write("<title>Mask R-CNN 查看器</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 20px; }\n")
            f.write(".sequence { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }\n")
            f.write("img { max-width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.2); margin-bottom: 10px; }\n")
            f.write("h2 { color: #333; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            f.write("<h1>Mask R-CNN 推理结果</h1>\n")
            
            f.write(f"<div class='sequence'>\n")
            f.write(f"<h2>图像 {idx}</h2>\n")
            
            # 六图拼接
            f.write(f"<h3>所有视图</h3>\n")
            f.write(f"<img src='1_六图拼接_{idx}.png'><br>\n")
            
            # 单独的图
            f.write(f"<h3>原图</h3>\n")
            f.write(f"<img src='2_原图_{idx}.png'><br>\n")
            
            f.write(f"<h3>检测框和类别</h3>\n")
            f.write(f"<img src='3_检测框和类别置信度_{idx}.png'><br>\n")
            
            f.write(f"<h3>只有mask</h3>\n")
            f.write(f"<img src='5_只有mask_{idx}.png'><br>\n")
            
            f.write(f"<h3>mask和原图</h3>\n")
            f.write(f"<img src='5_mask + 原图_{idx}.png'><br>\n")
            
            f.write(f"<h3>mask和类别</h3>\n")
            f.write(f"<img src='6_mask和类别_{idx}.png'><br>\n")
            
            f.write(f"<h3>mask、检测框和类别</h3>\n")
            f.write(f"<img src='7_mask框类别_{idx}.png'><br>\n")
            
            f.write("</div>\n")
            
            f.write("</body>\n</html>")
    
    # 生成主HTML索引页面
    main_html_path = os.path.join(date_dir, "index.html")
    with open(main_html_path, "w", encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head>\n")
        f.write('<meta charset="UTF-8">\n')
        f.write("<title>Mask R-CNN 结果索引</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 20px; }\n")
        f.write(".card { margin: 10px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; display: inline-block; width: 200px; text-align: center; }\n")
        f.write("a { text-decoration: none; color: #0066cc; }\n")
        f.write("a:hover { text-decoration: underline; }\n")
        f.write("h1 { color: #333; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>Mask R-CNN 推理结果索引</h1>\n")
        
        for idx, sequence_dir in results:
            rel_path = os.path.relpath(sequence_dir, date_dir)
            f.write(f"<div class='card'>\n")
            f.write(f"<h3>序列 {idx}</h3>\n")
            f.write(f"<a href='{rel_path}/index.html'>查看结果</a>\n")
            f.write("</div>\n")
        
        f.write("</body>\n</html>")
    
    return main_html_path

# ========= 启动服务器 =========
def start_http_server(date_dir):
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=date_dir, **kwargs)
        
        def end_headers(self):
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            super().end_headers()

    def run():
        server = HTTPServer(('0.0.0.0', 8080), CustomHandler)
        print(f"服务启动在 http://localhost:8080")
        print(f"结果保存在: {date_dir}")
        server.serve_forever()

    threading.Thread(target=run, daemon=True).start()

# ========= 主流程 =========
if __name__ == '__main__':
    print("输入图像编号（0 到 {}）用空格分隔，或直接回车使用默认序列：".format(len(dataset)-1))
    user_input = input("图像编号: ")
    if user_input.strip():
        ids = [int(i) for i in user_input.strip().split()]
    else:
        # ids = random.sample(range(len(dataset)), 5)
        # 使用较好 的数据：
        ids = [594, 3724, 2643, 3694, 3741, 1813, 3164]

    print(f"处理图像: {ids}")
    results, sequence_dirs = run_inference(ids)
    html_path = generate_html(results, sequence_dirs)
    start_http_server(date_dir)
    
    print("按Ctrl+C停止服务器")
    try:
        while True:
            input()
    except KeyboardInterrupt:
        print("服务器已停止")