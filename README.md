# Mask R-CNN 实例分割项目

本项目实现了基于Mask R-CNN的实例分割系统，包含两个版本的实现：基于torchvision的简化版本和完全自定义的PyTorch版本。

## 项目结构

```
.
├── mask-r-cnn/
│   ├── pytorch/                 # 自定义PyTorch实现
│   │   ├── config/              # 配置文件目录
│   │   │   └── coco_config.yaml # 主配置文件，定义数据路径、模型参数和训练设置
│   │   ├── dataset/             # 数据集处理相关代码
│   │   │   ├── __init__.py      # 包初始化文件
│   │   │   └── coco_dataset.py  # COCO数据集加载与预处理
│   │   ├── model/               # 模型定义目录
│   │   │   ├── backbone/        # 特征提取网络
│   │   │   │   ├── fpn.py       # 特征金字塔网络实现
│   │   │   │   └── resnet50.py  # ResNet50骨干网络实现
│   │   │   ├── heads/           # 模型头部定义
│   │   │   │   └── roi_head.py  # RoI处理头部，包含分类、边界框回归和掩码预测
│   │   │   ├── rcnn/            # R-CNN系列模型定义
│   │   │   │   ├── faster_rcnn.py  # Faster R-CNN实现
│   │   │   │   ├── general_rcnn.py # 通用R-CNN框架
│   │   │   │   └── mask_rcnn.py    # Mask R-CNN实现
│   │   │   ├── rpn/             # 区域提议网络
│   │   │   │   └── rpn.py       # RPN实现
│   │   │   ├── utils/           # 模型工具函数
│   │   │   │   ├── anchor.py    # 锚框生成
│   │   │   │   ├── boxes.py     # 边界框处理工具
│   │   │   │   ├── general.py   # 通用工具函数
│   │   │   │   ├── loss.py      # 损失函数定义
│   │   │   │   ├── misc_nn_ops.py # 其他神经网络操作
│   │   │   │   ├── poolers.py   # RoIAlign等池化操作
│   │   │   │   └── roi_align.py # RoIAlign实现
│   │   │   ├── __init__.py      # 模型包初始化
│   │   │   └── model_build.py   # 模型构建入口
│   │   ├── result/              # 结果输出目录
│   │   ├── train/               # 训练相关代码
│   │   │   ├── evaluate.py      # 模型评估代码
│   │   │   └── train.py         # 训练循环实现
│   │   ├── transforms/          # 数据变换和增强
│   │   │   └── mask_rcnn_transforms.py # 数据增强实现
│   │   ├── visualization/       # 可视化工具
│   │   │   ├── demo.py          # 批量图像预测和可视化
│   │   │   └── demo_one.py      # 单张图像预测和可视化
│   │   ├── __init__.py          # 包初始化
│   │   └── main.py              # 主入口文件
│   │
│   └── torchvision/             # 基于torchvision的实现
│       ├── config/              # 配置文件
│       ├── dataset/             # 数据集处理
│       ├── result/              # 结果输出
│       ├── transforms/          # 数据变换
│       ├── visualization/       # 可视化工具
│       └── main.py              # 主入口文件
│
└── 实验报告/                     # 实验报告目录
    ├── image/                   # 报告中使用的图片
    └── 实验报告-夏令营.md         # 详细的实验报告

```

## 两种实现的对比

### 1. torchvision实现 (mask-r-cnn/torchvision/)

- **特点**：使用torchvision.models.detection中预定义的Mask R-CNN模型
  - 代码简洁，实现快速
  - 使用经过优化的官方实现
  - 便于快速验证和部署
  不过，
  - 灵活性较低，难以修改内部结构
  - 对模型内部工作原理理解有限

### 2. 自定义PyTorch实现 (mask-r-cnn/pytorch/)

- **特点**：从头实现完整的Mask R-CNN架构
  - 完全掌控模型的每个组件
  - 高度灵活，便于修改和扩展
  - 深入理解模型工作原理
  - 便于研究和创新
 不过，
  - 代码量大，实现复杂
  - 需要更多的调试和验证


## 实验报告

详细的实验报告位于`实验报告/实验报告-夏令营.md`，包含了项目的完整说明、实现细节、实验结果和分析。 