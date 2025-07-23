#!/bin/bash

# Mask R-CNN训练示例脚本
# 展示不同的训练配置和参数使用方法

echo "========== Mask R-CNN 训练示例 =========="

# ==================== 示例1: 基础训练（使用默认参数） ====================
echo "示例1: 基础训练 - 使用所有默认参数"
echo "命令: python train.py"
echo "说明: 使用默认的COCO数据集路径、2个类别、5个epoch等"
echo ""

# ==================== 示例2: 自定义数据路径和类别数 ====================
echo "示例2: 自定义数据路径和类别数"
echo "命令: python train.py --data-path /path/to/your/coco --num-classes 81"
echo "说明: 指定自定义的COCO数据集路径，使用COCO的80个类别"
echo ""

# ==================== 示例3: 调整训练参数 ====================
echo "示例3: 调整训练参数"
echo "命令: python train.py --batch-size 4 --epochs 10 --lr 0.01"
echo "说明: 增大批次大小、延长训练轮数、提高学习率"
echo ""

# ==================== 示例4: 使用GPU并指定输出目录 ====================
echo "示例4: 指定设备和输出目录"
echo "命令: python train.py --device cuda --output-dir ./my_models"
echo "说明: 明确指定使用GPU训练，模型保存到指定目录"
echo ""

# ==================== 示例5: 从检查点恢复训练 ====================
echo "示例5: 从检查点恢复训练"
echo "命令: python train.py --resume ./result/model_epoch_3.pth --epochs 10"
echo "说明: 从第3个epoch的模型继续训练到第10个epoch"
echo ""

# ==================== 示例6: 仅测试模式 ====================
echo "示例6: 仅测试模式"
echo "命令: python train.py --test-only --resume ./result/model_final.pth"
echo "说明: 加载训练好的模型，仅在验证集上进行测试"
echo ""

# ==================== 示例7: 完整配置训练 ====================
echo "示例7: 完整配置训练"
cat << 'EOF'
python train.py \
    --data-path /datasets/coco \
    --num-classes 81 \
    --batch-size 8 \
    --epochs 26 \
    --lr 0.02 \
    --lr-milestones 16 22 \
    --weight-decay 1e-4 \
    --device cuda \
    --output-dir ./trained_models \
    --print-freq 20 \
    --save-freq 5
EOF
echo "说明: 类似于torchvision官方的完整训练配置"
echo ""

# ==================== 示例8: 小数据集快速实验 ====================
echo "示例8: 小数据集快速实验"
echo "命令: python train.py --epochs 2 --print-freq 5 --save-freq 1"
echo "说明: 快速验证代码功能，适合调试和小规模实验"
echo ""

# ==================== 示例9: CPU训练（无GPU环境） ====================
echo "示例9: CPU训练"
echo "命令: python train.py --device cpu --batch-size 1 --epochs 2"
echo "说明: 在没有GPU的环境下进行训练，需要调小批次大小"
echo ""

# ==================== 示例10: 自定义数据集路径 ====================
echo "示例10: 自定义数据集路径结构"
cat << 'EOF'
python train.py \
    --data-path /custom/dataset \
    --train-images images/train \
    --val-images images/val \
    --train-ann annotations/train.json \
    --val-ann annotations/val.json \
    --num-classes 5
EOF
echo "说明: 适用于自定义的数据集目录结构"
echo ""

echo "========== 参数说明 =========="
echo "查看所有可用参数: python train.py --help"
echo ""
echo "主要参数类别:"
echo "  数据相关: --data-path, --num-classes, --train-images, --val-images"
echo "  训练相关: --batch-size, --epochs, --lr, --momentum, --weight-decay"  
echo "  设备相关: --device, --output-dir"
echo "  其他功能: --resume, --test-only, --save-freq" 