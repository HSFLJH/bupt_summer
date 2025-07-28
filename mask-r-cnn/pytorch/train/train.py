from tqdm import tqdm
import math
import sys
import time

import torch
import utils.utils as utils

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, writer=None):
    """
    单轮训练主循环，集成tqdm进度条，实时显示训练进度。
    适合初学者理解的详细注释。
    """
    # 设置模型为训练模式（启用dropout、BN等）
    model.train()
    # 创建一个指标记录器，用于统计损失、学习率等
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 添加一个学习率的滑动窗口统计
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # 构造本轮训练的进度条标题
    header = f"Epoch: [{epoch}]"

    # 学习率预热调度器（只在第0个epoch用）
    lr_scheduler = None
    if epoch == 0:
        # 预热因子，刚开始时学习率较小，逐步增大
        warmup_factor = 1.0 / 1000
        # 预热步数，最多1000步或数据集长度-1
        warmup_iters = min(1000, len(data_loader) - 1)
        # 创建线性学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # tqdm包装数据加载器，显示进度条
    data_iter = tqdm(
        metric_logger.log_every(data_loader, print_freq, header),
        total=len(data_loader),
        desc=header,
        ncols=100
    )
    # 遍历每一个batch（小批量数据）
    for images, targets in data_iter:

        global_step = epoch * len(data_loader) + data_iter.n # 当前训练步数

        try:
            # 把所有图片移动到指定设备（如GPU）
            images = list(image.to(device) for image in images)
            # 把所有目标（标注）也移动到设备
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            # 自动混合精度训练AMP（节省显存，加速）
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # 前向传播，计算损失字典
                loss_dict = model(images, targets)
                # 总损失是所有损失项的和
                losses = sum(loss for loss in loss_dict.values())
        except Exception as e:
            print(f"跳过异常 batch: {e}")
            continue

        # 多卡训练时同步所有GPU的损失
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # 获取损失的数值
        loss_value = losses_reduced.item()

        # 如果损失不是有限数（如NaN或inf），停止训练
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # 梯度清零
        optimizer.zero_grad()
        if scaler is not None:
            # 混合精度反向传播和优化
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准反向传播和优化
            losses.backward()
            optimizer.step()

        # 如果有预热调度器，更新学习率
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 更新指标记录器（损失、学习率等）
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 写入tensorboard日志

        if writer:
            writer.add_scalar("Loss/Total", loss_value, global_step)
            writer.add_scalar("Loss/Classifier", loss_dict_reduced['loss_classifier'].item(), global_step)
            writer.add_scalar("Loss/Box", loss_dict_reduced['loss_box_reg'].item(), global_step)
            writer.add_scalar("Loss/Mask", loss_dict_reduced['loss_mask'].item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)

        # 在tqdm进度条上显示当前loss和lr
        data_iter.set_postfix({
            'loss': f'{losses_reduced.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    # 返回本轮训练的指标统计
    return metric_logger
