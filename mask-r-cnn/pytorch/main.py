"""
Mask R-CNN主程序入口  
"""

import os
import argparse
import yaml
import torch
import random
import numpy as np
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

from dataset.coco_dataset import create_coco_dataloader
from visualization.dataset_visualize import preview_augmentations
from model.model_build import get_model_mask_r_cnn
from train.train import train_one_epoch
from train.evaluate import evaluate


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Mask R-CNN 训练与评估")
    
    # 配置文件
    parser.add_argument('--config', type=str, default='/home/lishengjie/study/sum_hansf/bupt_summer/mask-r-cnn/pytorch/config/coco_config.yaml',
                        help='配置文件路径')
    
    # 模式选择
    parser.add_argument('--train', action='store_true', default=False,
                        help='训练模式')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='评估模式')
    
    # 数据增强预览
    parser.add_argument('--augmentation-preview', action='store_true', default=False,
                        help='预览数据增强效果')
    parser.add_argument('--num-preview-samples', type=int, default=5,
                        help='预览的样本数量')
    parser.add_argument('--sample-indices', type=str, default='',
                        help='指定要预览的样本索引，用逗号分隔，例如"1,2,3"。如果提供此参数，则忽略num-preview-samples')
    parser.add_argument('--show-annotations', action='store_true', default=True,
                        help='是否显示标注（边界框和掩码）')
    
    # 运行demo
    parser.add_argument('--demo', action='store_true', default=False,
                        help='运行demo')
    parser.add_argument('--demo-one', action='store_true', default=False,
                        help='运行demo_one')
    
    # 路径
    parser.add_argument('--output-dir', type=str, default='/home/lishengjie/study/sum_hansf/bupt_summer/mask-r-cnn/pytorch/result',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    
    # 其他参数
    parser.add_argument('--workers', type=int, default=4,
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
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    return config


def main():
    """
    主函数
    """

    # ============================== 读取配置文件 ==============================

    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = setup_config(args)
    
    # 强制使用CUDA，不再需要setup_environment函数和随机种子
    if not torch.cuda.is_available():
        raise RuntimeError("错误: 此项目强制要求使用CUDA，但未检测到可用GPU。")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # export CUDA_VISIBLE_DEVICES=3

    # 开启CUDNN性能基准测试以提升速度
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda")
    print(f"强制使用CUDA: {torch.cuda.get_device_name(0)}")

    # ============================== 运行demo ==============================
    if args.demo:
        from visualization.demo import main as demo_main
        demo_main()
    elif args.demo_one:
        from visualization.demo_one import main as demo_one_main
        demo_one_main()

    # ============================== 加载数据集 ==============================
    
    # 加载数据集
    train_dataset, train_loader = create_coco_dataloader(config, train=True)
    val_dataset, val_loader = create_coco_dataloader(config, train=False)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # ============================== 数据增强预览 ==============================
    
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
    
    # ==================== 【模型构建】 ====================
    print("========== 模型构建开始 ==========")
    model = get_model_mask_r_cnn(config)
    model.to(device)
    print(f"模型类别数: {config['model']['num_classes']}")

    # ==================== 【优化器与学习率调度】 ====================
    print("========== 优化器设置开始 ==========")
    # 优化器设置：只优化需要梯度的参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['lr_milestones'],
        gamma=config['training']['lr_gamma']
    )
    print(f"初始学习率: {config['training']['lr']}")
    print(f"学习率调度: 在epoch {config['training']['lr_milestones']} 时衰减为 {config['training']['lr_gamma']} 倍")

    # ==================== 【模型恢复】 ====================
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载模型权重: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"警告: 无法找到模型文件 {args.resume}")
    
    # ==================== 【输出目录和日志】 ====================
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, f"maskrcnn_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    print(f"模型和日志将保存到: {output_dir}")

    # ==================== 【仅评估模式】 ====================
    if args.eval:
        print("========== 仅评估模式 ==========")
        if not args.resume:
            print("错误: 评估模式需要通过 --resume 指定模型路径")
            return
        evaluate(model, val_loader, device=device)
        return

    # ==================== 【训练循环】 ====================
    if args.train:
        print("========== 开始训练 ==========")
        print(f"训练参数:")
        print(f"  - 批次大小: {config['training']['batch_size']}")
        print(f"  - 训练轮数: {config['training']['epochs']}")
        print(f"  - 设备: {device}")
        print(f"  - 输出目录: {output_dir}")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, config['training']['epochs']):
            print(f"\n========== Epoch {epoch+1}/{config['training']['epochs']} ==========")
            
            # 训练一个epoch
            train_one_epoch(model, optimizer, train_loader, device, epoch, config['training']['print_freq'], writer=writer)
            
            # 更新学习率
            lr_scheduler.step()
            
            # 在验证集上评估
            evaluator = evaluate(model, val_loader, device=device)

            # 将评估结果写入TensorBoard
            if writer and evaluator and evaluator.coco_eval:
                if "bbox" in evaluator.coco_eval:
                    stats = evaluator.coco_eval["bbox"].stats
                    writer.add_scalar("Val/bbox_mAP", stats[0], epoch)
                    writer.add_scalar("Val/bbox_mAP50", stats[1], epoch)
                    writer.add_scalar("Val/bbox_mAP75", stats[2], epoch)

                if "segm" in evaluator.coco_eval:
                    stats = evaluator.coco_eval["segm"].stats
                    writer.add_scalar("Val/mask_mAP", stats[0], epoch)
                    writer.add_scalar("Val/mask_mAP50", stats[1], epoch)
                    writer.add_scalar("Val/mask_mAP75", stats[2], epoch)

            # 保存检查点
            if (epoch + 1) % config['training']['save_freq'] == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'config': config
                }, checkpoint_path)
                print(f"检查点已保存: {checkpoint_path}")

        # 训练结束
        total_time = time.time() - start_time
        print(f"\n========== 训练完成！ ==========")
        print(f"总用时: {str(datetime.timedelta(seconds=int(total_time)))}")
        print(f"最终模型和日志保存在: {output_dir}")
    
    writer.close()
    
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
