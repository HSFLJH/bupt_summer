# Mask R-CNN 实现

本项目实现了 Mask R-CNN 模型用于实例分割任务，特别关注了数据加载和数据增强部分。

## 项目结构

```
mask r-cnn/pytorch/
├── config/                  # 配置文件
│   └── coco_config.yaml     # COCO数据集配置
├── dataset/                 # 数据集加载
│   └── coco_dataset.py      # COCO数据集加载器
├── transforms/              # 数据增强和转换
│   └── mask_rcnn_transforms.py  # Mask R-CNN数据增强实现
├── visualization/           # 可视化工具
│   └── visualize.py         # 数据增强和结果可视化
├── main.py                  # 主程序入口
└── README.md                # 本文档
```

## 数据增强实现

我们为Mask R-CNN实现了一系列适合实例分割任务的数据增强方法：

1. **Large Scale Jitter (LSJ)**：大尺度抖动，随机缩放图像并相应地调整边界框和掩码。
2. **Random Horizontal Flip**：随机水平翻转图像，同时处理边界框和掩码。
3. **Color Jitter**：随机调整图像的亮度、对比度、饱和度和色调。
4. **Random Grayscale**：随机将图像转换为灰度图，提高对颜色变化的鲁棒性。
5. **Small Rotation**：小角度旋转图像（±10度），同时更新掩码和边界框。
6. **Safe Random Crop**：安全裁剪，确保实例的完整性（至少保留指定比例的实例面积）。
7. **Motion Blur**：运动模糊，模拟相机或物体运动导致的模糊效果，增强模型对模糊图像的识别能力。
8. **Random Perspective**：随机透视变换，模拟不同视角下的物体外观，提高模型对视角变化的鲁棒性。
9. **Normalization**：使用ImageNet预训练模型的均值和标准差进行标准化。

这些方法特别考虑了实例分割任务的特殊性，确保在变换图像的同时正确变换相应的边界框和分割掩码。

## 数据加载

COCO数据集加载器实现了以下功能：

1. 使用 `pycocotools` 加载和解析COCO格式的注释。
2. 处理图像、边界框、类别标签和实例掩码。
3. 跳过无效的标注（太小的边界框、crowd区域等）。
4. 将COCO类别ID映射到连续的类别ID。
5. 应用配置的数据增强管道。

## collate_fn 函数

对于目标检测和实例分割任务，每个图像可能包含不同数量的目标，标准的批处理方法不适用。我们的 `collate_fn` 函数：

1. 保持每个样本的图像和标注信息的对应关系。
2. 不进行填充或堆叠，而是将图像和标注分别组合成元组。
3. 在模型前向传播时，可以单独处理每个样本，然后汇总损失。

## 使用方法

1. 安装依赖：PyTorch、torchvision、pycocotools、opencv-python、matplotlib

2. 修改配置文件：`config/coco_config.yaml`，设置数据集路径等参数。

3. 运行数据增强预览：
   ```
   python main.py --augmentation-preview
   ```

4. 训练模型（待实现）：
   ```
   python main.py --train
   ```

5. 评估模型（待实现）：
   ```
   python main.py --eval
   ``` 