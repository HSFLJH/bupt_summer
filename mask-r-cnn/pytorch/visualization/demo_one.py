import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from pycocotools.coco import COCO
from visualization.dataset_coco import CocoInstanceDataset, get_transform
from model.model_build import get_model_mask_r_cnn
import numpy as np

def load_category_names(ann_file):
    """
    从 COCO 标注文件中读取类别名称，返回字典和列表
    """
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {cat['id']: cat['name'] for cat in cats}
    id_list = [id_to_name[i] for i in sorted(id_to_name)]
    return id_to_name, id_list


def visualize_prediction(img_tensor, prediction, id_to_name, save_path=None, score_threshold=0.5):
    """
    可视化预测结果:显示边框、标签、mask
    """
    #图像与处理
    plt.figure(figsize=(10, 10), facecolor='black')

    # 1. 准备好要显示的底层图片 (H, W, C)
    img_to_show = img_tensor.permute(1, 2, 0).cpu().numpy()

    ax = plt.gca()

    # 2. 创建一个空白的、和原图一样大的图层，用来画所有的 masks
    # 这个图层将一次性地叠加在原图上
    masks_overlay = np.zeros_like(img_to_show)
    
    masks = prediction['masks']
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    num_instances = len(masks)
    colors = plt.get_cmap('hsv', num_instances)#使用 HSV 颜色映射为每个实例分配不同颜色
    #置信度过滤
    for i in range(num_instances):
        score = scores[i].item()
        if score < score_threshold:
            continue

        color = colors(i)
        
        # 3. 在空白图层上填充 mask 的颜色，不在这里 imshow
        mask = masks[i, 0].cpu().numpy() > 0.5
        masks_overlay[mask] = color[:3] # 只填充 RGB 颜色

        # 边框和文字的绘制逻辑不变
        box = boxes[i].cpu().numpy()
        label_id = labels[i].item()
        label_name = id_to_name.get(label_id, str(label_id))
        # 绘制边框
        ax.add_patch(patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                      linewidth=2, edgecolor=color, facecolor='none'))
        #在边界框上方显示类别名称和置信度分数
        ax.text(box[0], box[1] - 5, f'{label_name}: {score:.2f}', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.8), color='white')

    # 4. 所有 mask 都画到空白图层上之后，最后一次性地、半透明地将这个图层叠加到原图上
    plt.imshow(masks_overlay)

    # 5. 最后画上原图
    plt.imshow(img_to_show, alpha=0.7) # 画好mask之后，放上图片，mask是透明的

    plt.axis('off')
    if save_path:
        # 使用 pad_inches=0 来去除白边
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, facecolor='black')
        print(f"已保存结果至: {save_path}")
        plt.close()
    else:
        plt.show()

def run_inference(model_path, img_folder, ann_file, img_indices, save_dir, score_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载类别映射
    id_to_name, _ = load_category_names(ann_file)

    # 加载模型
    model = get_model_mask_r_cnn(num_classes=91)
    assert os.path.exists(model_path), f"模型路径不存在: {model_path}"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载数据
    dataset = CocoInstanceDataset(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=get_transform(train=False)
    )

    os.makedirs(save_dir, exist_ok=True)

    for idx in img_indices:
        img, _ = dataset[idx]
        with torch.no_grad():
            prediction = model([img.to(device)])[0]
        save_path = os.path.join(save_dir, f"inference_result_{idx}.png")
        visualize_prediction(img, prediction, id_to_name, save_path=save_path, score_threshold=score_threshold)


def parse_indices(indices_str):
    """
    解析图像索引，支持逗号分隔和空格分隔
    """
    if isinstance(indices_str, list):
        return indices_str  # 如果已经是列表，直接返回
    
    if ',' in indices_str:
        # 处理逗号分隔的情况
        return [int(idx.strip()) for idx in indices_str.split(',')]
    else:
        # 处理空格分隔的情况
        return [int(indices_str)]


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN 推理与可视化（增强版）")
    parser.add_argument("--model-path", type=str, help="训练好的模型权重路径",
                        default="/home/lishengjie/study/sum_hansf/bupt_summer/mask-r-cnn/pytorch/result/model/model1.pth")
    parser.add_argument("--img-folder", type=str, help="验证集图像目录",
                        default="/home/lishengjie/data/COCO2017/val2017")
    parser.add_argument("--ann-file", type=str, help="COCO标注文件路径",
                        default="/home/lishengjie/data/COCO2017/annotations/instances_val2017.json")
    parser.add_argument("--save-dir", type=str, help="可视化结果保存目录",
                        default="/home/lishengjie/study/sum_hansf/bupt_summer/mask-r-cnn/pytorch/result/result_pngs/one")
    parser.add_argument("--indices", type=str, default="594", 
                        help="图像索引，可用逗号分隔输入多个，例如：594,1813,3724")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="置信度阈值，低于此值不显示")

    args = parser.parse_args()
    
    # 解析图像索引
    img_indices = parse_indices(args.indices)
    print(f"处理图像索引: {img_indices}")

    run_inference(
        model_path=args.model_path,
        img_folder=args.img_folder,
        ann_file=args.ann_file,
        img_indices=img_indices,
        save_dir=args.save_dir,
        score_threshold=args.score_thresh
    )

if __name__ == "__main__":
    main()
