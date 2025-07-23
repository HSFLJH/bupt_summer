# 主训练脚本 - Mask R-CNN训练流程
# 精简版本，专注于核心训练逻辑，适合学习和快速原型开发
# 支持命令行参数配置，灵活性更强

import os
import time
import datetime
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dataset_coco import CocoInstanceDataset, get_transform
from model import get_instance_segmentation_model
from engine import train_one_epoch, evaluate
from utils import collate_fn, mkdir


def parse_args():
    """
    解析命令行参数
    
    返回:
        args: 包含所有配置参数的命名空间
    """
    parser = argparse.ArgumentParser(description='Mask R-CNN训练脚本 - 支持自定义参数配置')
    
    # ==================== 【数据相关参数】 ====================
    parser.add_argument('--data-path', default='/root/autodl-tmp/COCO', type=str, 
                       help='COCO数据集根目录路径 ')
    parser.add_argument('--train-images', default='train2017', type=str,
                       help='训练图像文件夹名称 ')
    parser.add_argument('--val-images', default='val2017', type=str,
                       help='验证图像文件夹名称 ')
    parser.add_argument('--train-ann', default='annotations/instances_train2017.json', type=str,
                       help='训练标注文件路径 ')
    parser.add_argument('--val-ann', default='annotations/instances_val2017.json', type=str,
                       help='验证标注文件路径')
    
    # ==================== 【模型相关参数】 ====================
    parser.add_argument('--num-classes', default=91, type=int,
                       help='类别数量（包括背景类） (默认: 2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='是否使用预训练权重 (默认: True)')
    
    # ==================== 【训练相关参数】 ====================
    parser.add_argument('--batch-size', default=4, type=int,
                       help='批次大小 (默认: 4)')
    parser.add_argument('--epochs', default=3, type=int,
                       help='训练轮数 (默认: 3)')
    parser.add_argument('--lr', default=0.005, type=float,
                       help='初始学习率 (默认: 0.005)')
    parser.add_argument('--momentum', default=0.9, type=float,
                       help='SGD动量参数 (默认: 0.9)')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                       help='权重衰减系数 (默认: 1e-4)')
    parser.add_argument('--lr-milestones', default=[3, 4], nargs='+', type=int,
                       help='学习率调度里程碑 (默认: [3, 4])')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                       help='学习率衰减因子 (默认: 0.1)')
    
    # ==================== 【设备和输出参数】 ====================
    parser.add_argument('--device', default='cuda', type=str,
                       help='训练设备 (cuda/cpu，默认: cuda)')
    parser.add_argument('--output-dir', default='./result', type=str,
                       help='模型保存路径 (默认: ./result)')
    parser.add_argument('--print-freq', default=10, type=int,
                       help='训练日志打印频率 (默认: 10)')
    
    # ==================== 【其他参数】 ====================
    parser.add_argument('--resume', default='', type=str,
                       help='恢复训练的模型路径 (默认: 空，从头开始训练)')
    parser.add_argument('--test-only', action='store_true',
                       help='仅进行测试，不训练模型')
    parser.add_argument('--save-freq', default=1, type=int,
                       help='模型保存频率（每多少个epoch保存一次） (默认: 1)')
    
    return parser.parse_args()


def main(args):
    """
    主训练函数
    
    参数:
        args: 命令行参数
    
    训练流程：
    1. 设置训练参数和环境
    2. 加载数据集
    3. 构建模型
    4. 设置优化器和学习率调度
    5. 执行训练循环
    """

    # ==================== 【参数设置】 ====================
    # 设备配置
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU训练: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用CPU训练")
    
    # 创建输出目录，按时间戳命名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"maskrcnn_{timestamp}")
    mkdir(output_dir)
    print(f"模型将保存到: {output_dir}")

    # ==================== 【数据加载与预处理】 ====================
    print("========== 数据加载开始 ==========")
    
    # 1. 【训练集加载】
    train_img_folder = os.path.join(args.data_path, args.train_images)
    train_ann_file = os.path.join(args.data_path, args.train_ann)
    
    dataset = CocoInstanceDataset(
        img_folder=train_img_folder,                                        # 训练图像路径
        ann_file=train_ann_file,                                           # 训练标注文件
        transforms=get_transform(train=True),                              # 训练时数据变换（包含数据增强）
    )
    print(f"训练集大小: {len(dataset)} 张图像")
    
    # 2. 【验证集加载】
    val_img_folder = os.path.join(args.data_path, args.val_images)
    val_ann_file = os.path.join(args.data_path, args.val_ann)
    
    dataset_test = CocoInstanceDataset(
        img_folder=val_img_folder,                                         # 验证图像路径
        ann_file=val_ann_file,                                            # 验证标注文件
        transforms=get_transform(train=False),                            # 验证时数据变换（无数据增强）
    )
    print(f"验证集大小: {len(dataset_test)} 张图像")

    # 3. 【数据采样器设置】
    train_sampler = RandomSampler(dataset)        # 训练时随机采样
    test_sampler = SequentialSampler(dataset_test) # 验证时顺序采样

    # 4. 【数据加载器设置】
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        collate_fn=collate_fn,
        num_workers=8,  # 多进程加载
        pin_memory=True  # 加速GPU传输
    )
    data_loader_test = DataLoader(
        dataset_test, 
        batch_size=1,  # 验证时使用batch_size=1
        sampler=test_sampler, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # ==================== 【模型构建】 ====================
    print("========== 模型构建开始 ==========")
    model = get_instance_segmentation_model(num_classes=args.num_classes)
    model.to(device)  # 将模型移动到GPU/CPU
    print(f"模型类别数: {args.num_classes}")
    
    # 【模型恢复】如果指定了resume路径
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载模型权重: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint)
        else:
            print(f"警告: 无法找到模型文件 {args.resume}")

    # ==================== 【优化器与学习率调度】 ====================
    print("========== 优化器设置开始 ==========")
    # 优化器设置：只优化需要梯度的参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=args.lr_milestones, 
        gamma=args.lr_gamma
    )
    
    print(f"初始学习率: {args.lr}")
    print(f"学习率调度: 在epoch {args.lr_milestones} 时衰减为 {args.lr_gamma} 倍")

    # ==================== 【仅测试模式】 ====================
    if args.test_only:
        print("========== 仅测试模式 ==========")
        evaluate(model, data_loader_test, device=device)
        return

    # ==================== 【训练循环】 ====================
    print("========== 开始训练 ==========")
    print(f"训练参数:")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 设备: {device}")
    print(f"  - 输出目录: {output_dir}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n========== Epoch {epoch+1}/{args.epochs} ==========")
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        # 【单轮训练】训练一个epoch
        train_one_epoch(
            model, 
            optimizer, 
            data_loader, 
            device, 
            epoch, 
            print_freq=args.print_freq
        )
        
        # 【学习率更新】
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 【模型保存】根据保存频率保存模型权重
        if (epoch + 1) % args.save_freq == 0:
            model_save_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存: {model_save_path}")\

        # 【模型评估】在验证集上评估模型性能
        print("开始验证...")
        evaluate(model, data_loader_test, device=device)

    # 【最终模型保存】
    final_model_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存: {final_model_path}")

    # ==================== 【训练完成】 ====================
    total_time = time.time() - start_time
    print(f"\n========== 训练完成！ ==========")
    print(f"总用时: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"模型保存位置: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
