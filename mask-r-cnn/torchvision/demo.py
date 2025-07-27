import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from PIL import Image
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading

# ========== 配置路径与环境 ==========
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
output_dir = "/mnt/data/mask_rcnn_inference_output"
os.makedirs(output_dir, exist_ok=True)

# ========== 模拟数据与模型 ==========
# 以下为模拟图像和预测结果（请在真实运行时替换为真实模型和 COCO 数据集）
def get_mock_image():
    img = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)
    return img

def get_mock_prediction():
    boxes = torch.tensor([[30, 30, 100, 100], [120, 60, 200, 160]], dtype=torch.float32)
    labels = torch.tensor([1, 2])
    scores = torch.tensor([0.95, 0.88])
    masks = torch.randint(0, 2, (2, 256, 256), dtype=torch.uint8)
    return {"boxes": boxes, "labels": labels, "scores": scores, "masks": masks}

# ========== 可视化函数 ==========
def visualize(image, prediction):
    masks = prediction["masks"]
    labels = prediction["labels"]
    scores = prediction["scores"]
    boxes = prediction["boxes"]

    subfigs = []

    # 原图
    subfigs.append(to_pil_image(image))

    # 检测框 + 类别
    box_img = draw_bounding_boxes(image, boxes, [str(l.item()) for l in labels], colors="red")
    subfigs.append(to_pil_image(box_img))

    # 检测框 + 类别 + 分数
    box_score_img = draw_bounding_boxes(
        image, boxes,
        [f"{l.item()}:{s:.2f}" for l, s in zip(labels, scores)],
        colors="green"
    )
    subfigs.append(to_pil_image(box_score_img))

    # mask 叠加（无边框）
    mask_only = image.clone()
    for m in masks:
        mask_only[0][m.bool()] = 255
    subfigs.append(to_pil_image(mask_only))

    # mask + label
    mask_label = image.clone()
    for m, l in zip(masks, labels):
        mask_label[1][m.bool()] = 255
    subfigs.append(to_pil_image(mask_label))

    # mask + box + label
    mix_img = image.clone()
    for m in masks:
        mix_img[2][m.bool()] = 255
    mix_img_draw = draw_bounding_boxes(
        mix_img, boxes,
        [str(l.item()) for l in labels],
        colors="blue"
    )
    subfigs.append(to_pil_image(mix_img_draw))

    return subfigs

# ========== 主流程 ==========
sequence_paths = []
for i in range(5):
    img = get_mock_image()
    pred = get_mock_prediction()
    images = visualize(img, pred)

    fig, axs = plt.subplots(1, 6, figsize=(20, 5))
    for j in range(6):
        axs[j].imshow(images[j])
        axs[j].set_title(f"View {j+1}")
        axs[j].axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"sequence_{i+1}.png")
    plt.savefig(save_path)
    plt.close()
    sequence_paths.append(f"sequence_{i+1}.png")

# ========== HTML 页面生成 ==========
html_path = os.path.join(output_dir, "index.html")
with open(html_path, "w") as f:
    f.write("<html><head><title>Mask R-CNN Inference Viewer</title></head><body>")
    f.write("<h1>Mask R-CNN Inference Results</h1>")
    for img_name in sequence_paths:
        f.write(f"<div style='margin-bottom:50px;'>")
        f.write(f"<h2>{img_name}</h2>")
        f.write(f"<img src='{img_name}' width='100%'><br>")
        f.write(f"</div>")
    f.write("</body></html>")

# ========== 启动HTTP服务器 ==========
class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=output_dir, **kwargs)

def run_server():
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, CustomHandler)
    print("Serving at http://localhost:8080")
    httpd.serve_forever()

# 使用线程后台运行服务器
thread = threading.Thread(target=run_server, daemon=True)
thread.start()
