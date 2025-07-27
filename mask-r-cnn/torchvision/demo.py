import os
import random
import torch
import matplotlib.pyplot as plt
from dataset_coco import CocoInstanceDataset, get_transform
from model import get_instance_segmentation_model

# ==================== 【环境设置】 ====================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ==================== 【模型加载】 ====================
# 【模型初始化】创建与训练时相同的模型结构
model = get_instance_segmentation_model(num_classes=2)

# 【权重加载】加载训练好的模型权重
model.load_state_dict(torch.load("maskrcnn_coco.pth"))
model.to(device)
model.eval()  # 设置为评估模式，关闭dropout和batch normalization

# ==================== 【测试数据准备】 ====================
# 【数据集初始化】加载验证集用于测试
dataset = CocoInstanceDataset(
    img_folder="coco/val2017",                                    # 验证图像路径
    ann_file="coco/annotations/instances_val2017.json",          # 验证标注文件
    transforms=get_transform()                                   # 图像预处理变换
)

# 【单张图像获取】获取数据集中的第一张图像
img, _ = dataset[0]

# ==================== 【模型推理】 ====================
# 【推理过程】关闭梯度计算以节省内存和加速推理
with torch.no_grad():
    # 输入需要是列表格式，即使只有一张图像
    prediction = model([img.to(device)])

# ==================== 【结果可视化】 ====================
# 【图像格式转换】将张量转换为可显示的numpy数组
# 从[0,1]范围转换到[0,255]，并调整维度顺序从CHW到HWC
img_show = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()

# 【显示原图】
plt.imshow(img_show)

# 【mask叠加显示】遍历所有检测到的实例，显示分割mask
for i in range(len(prediction[0]['masks'])):
    # 【mask处理】提取单个mask并转换为可显示格式
    mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
    
    # 【半透明叠加】将mask以半透明方式叠加到原图上
    plt.imshow(mask, alpha=0.4)

# 【图像显示设置】
plt.axis('off')  # 关闭坐标轴显示
plt.show()       # 显示最终结果
