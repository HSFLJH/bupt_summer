"""
Mask R-CNN主程序入口
"""

import os
import argparse
import yaml
import torch
import random
import numpy as np
from dataset.coco_dataset import create_coco_dataloader
from visualization.dataset_visualize import preview_augmentations


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Mask R-CNN 训练与评估")
    
    # 配置文件
    parser.add_argument('--config', type=str, default='/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/pytorch/config/coco_config.yaml',
                        help='配置文件路径')
    
    # 模式选择
    parser.add_argument('--train', action='store_true',
                        help='训练模式')
    parser.add_argument('--eval', action='store_true',
                        help='评估模式')
    
    # 数据增强预览
    parser.add_argument('--augmentation-preview', action='store_true', default=True,
                        help='预览数据增强效果')
    parser.add_argument('--num-preview-samples', type=int, default=5,
                        help='预览的样本数量')
    parser.add_argument('--sample-indices', type=str, default='',
                        help='指定要预览的样本索引，用逗号分隔，例如"1,2,3"。如果提供此参数，则忽略num-preview-samples')
    parser.add_argument('--show-annotations', action='store_true', default=True,
                        help='是否显示标注（边界框和掩码）')
    parser.add_argument('--augmentation-level', type=int, default=1, choices=[1, 2, 3, 4],
                        help='数据增强级别 (0-4): 0=无, 1=基础（实例分割推荐）, 2=默认, 3=较强, 4=最强')
    
    # 路径
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--workers', type=int, default=None,
                        help='数据加载器工作进程数，覆盖配置文件中的设置')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批处理大小，覆盖配置文件中的设置')
    
    return parser.parse_args()


def setup_config(args):
    """
    读取并设置配置
    
    Args:
        args: 命令行参数
    
    Returns:
        config: 配置字典
    """
    # 读取配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置文件
    if args.workers is not None:
        config['training']['workers'] = args.workers
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # 数据增强级别
    if args.augmentation_level is not None:
        config['transforms']['augmentation_level'] = args.augmentation_level
        print(f"使用数据增强级别: {args.augmentation_level}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    return config


def setup_environment(args, config):
    """
    设置环境，包括随机种子、CUDA等
    
    Args:
        args: 命令行参数
        config: 配置字典
    """
    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 设置CUDA环境
    if config['device']['use_cuda'] and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # 如果指定了可见设备
        if 'cuda_visible_devices' in config['device']:
            os.environ["CUDA_VISIBLE_DEVICES"] = config['device']['cuda_visible_devices']
        
        # 确保CUDA运算是确定性的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        device = torch.device("cuda")
        print(f"使用CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    return device


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = setup_config(args)
    
    # 设置环境
    device = setup_environment(args, config)
    
    # 加载数据集
    train_dataset, train_loader = create_coco_dataloader(config, train=True)
    val_dataset, val_loader = create_coco_dataloader(config, train=False)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 如果是增强预览模式，则运行预览并退出
    if args.augmentation_preview:
        print("预览数据增强效果...")
        sample_indices = None
        if args.sample_indices:
            try:
                sample_indices = [int(idx) for idx in args.sample_indices.split(',')]
                print(f"使用指定的样本索引: {sample_indices}")
            except ValueError:
                print("样本索引格式错误，将使用随机样本")
        
        preview_augmentations(
            val_dataset, 
            args.num_preview_samples, 
            config, 
            sample_indices,
            show_annotations=args.show_annotations
        )
        return
    
    # TODO: 加载模型、优化器等
    # TODO: 训练或评估模型
    
    print("暂未实现模型训练和评估部分。")


if __name__ == "__main__":
    """
    collate_fn函数解释：
    
    在PyTorch的DataLoader中，collate_fn用于将多个样本组合成一个批次。
    对于Mask R-CNN等目标检测/实例分割任务，由于每个图像可能有不同数量的目标，
    标准的批处理无法直接应用。
    
    我们的collate_fn函数(在mask_rcnn_transforms.py中实现)做了以下工作：
    1. 保持每个样本的图像和标注信息的对应关系
    2. 不进行填充或堆叠，而是将图像和标注分别组合成元组
    3. 在模型前向传播时，可以单独处理每个样本，然后汇总损失
    
    返回格式为：(images_tuple, targets_tuple)
    - images_tuple: 包含批次中所有图像的元组
    - targets_tuple: 包含批次中所有目标标注字典的元组
    """
    main()
