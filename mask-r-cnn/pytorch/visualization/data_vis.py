import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import shutil

def load_tensorboard_data(log_dir):
    """
    加载并返回 TensorBoard 事件累加器对象。
    """
    # 查找最新的事件文件
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        print(f"目录 {log_dir} 中未找到 TensorBoard 事件日志文件")
        return None
    
    # 通常最新的文件是我们需要分析的
    event_path = os.path.join(log_dir, sorted(event_files)[-1])
    
    print(f"正在从 {event_path} 加载数据...")
    # 加载事件数据
    ea = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={  # 设置加载数据的规则，_scalars_ 表示所有标量数据
            event_accumulator.SCALARS: 0,
        },
    )
    ea.Reload()  # 加载所有数据
    return ea


def plot_individual_metrics(ea, output_dir):
    """
    为事件文件中的每一个标量指标绘制并保存一张独立的图表。
    
    参数:
        ea: 事件累加器对象 (EventAccumulator)
        output_dir: 保存图像的目录
    """
    print("\n--- 开始绘制独立指标图表 ---")
    # 获取所有可用的标量 'tags'
    all_tags = ea.Tags().get("scalars", [])
    if not all_tags:
        print("未找到任何标量数据。")
        return

    for tag in all_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        plt.figure(figsize=(12, 7))
        plt.plot(steps, values, label=tag, color='dodgerblue')
        
        # 使用 tag 作为英文标题
        title = tag.replace('/', ' - ')
        plt.title(f"Metric: {title}", fontsize=16)
        plt.xlabel("Epoch / Step", fontsize=12)
        plt.ylabel("Value", fontsize=12) # <--- 修改为英文
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 创建一个有效的文件名，替换特殊字符
        output_filename = f"{tag.replace('/', '_')}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close() # 关闭当前图像，防止在内存中累积
        print(f"图表已保存: {output_path}")

def plot_combined_metrics(ea, tags, title, output_path):
    """
    将指定的多个指标绘制在同一张图表上。
    
    参数:
        ea: 事件累加器对象 (EventAccumulator)
        tags: 要在同一张图上绘制的标量名称列表
        title: 图表标题 (英文)
        output_path: 完整的图像保存路径
    """
    print(f"\n--- 开始绘制聚合图表: {title} ---")
    plt.figure(figsize=(12, 7))
    
    has_data = False
    for tag in tags:
        if tag not in ea.Tags().get("scalars", []):
            print(f"警告: 指标 {tag} 不存在于日志中，已跳过。")
            continue
        
        has_data = True
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, label=tag)
    
    # 如果没有任何一条线被画出，就不保存空图片
    if not has_data:
        print(f"聚合图表 '{title}' 中没有任何有效数据，已跳过生成图片。")
        plt.close()
        return

    plt.title(title, fontsize=16)
    plt.xlabel("Epoch / Step", fontsize=12)
    plt.ylabel("Value", fontsize=12) # <--- 修改为英文
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12) # 显示图例以区分不同曲线
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"聚合图表已保存: {output_path}")


if __name__ == "__main__":
    # --- 1. 参数配置 ---
    log_dir = "/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/torchvision/result/three/tensorboard"  # 日志目录
    output_dir = "/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/torchvision/result/three/train_pngs" # 保存路径

    # 如果输出目录已存在，可以选择清空它，避免旧图片干扰
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"输出目录已创建: {output_dir}")

    # --- 2. 加载数据 ---
    event_accumulator_data = load_tensorboard_data(log_dir)

    if event_accumulator_data:
        # --- 3. 绘制所有独立的指标图 ---
        plot_individual_metrics(event_accumulator_data, output_dir)
        
        # --- 4. 定义要聚合的指标并绘制聚合图 ---
        
        # 定义 Loss 相关指标
        loss_tags = [
            "Loss/Total", 
            "Loss/Classifier", 
            "Loss/Box", 
            "Loss/Mask",
            "Loss/Objectness", # RPN网络的损失
            "Loss/RPN Box"   # RPN网络的回归损失
        ]
        
        # 定义 mAP 相关指标
        map_tags = [
            "Val/bbox_mAP",
            "Val/bbox_mAP50",
            "Val/bbox_mAP75",
            "Val/mask_mAP",
            "Val/mask_mAP50",
            "Val/mask_mAP75",
        ]
        
        # 定义其他你可能关心的指标，比如学习率
        other_tags = ["LR"]

        # 绘制 Loss 聚合图
        plot_combined_metrics(
            ea=event_accumulator_data, 
            tags=loss_tags, 
            title="Training Loss Curves", # 
            output_path=os.path.join(output_dir, "combined_losses.png")
        )
        
        # 绘制 mAP 聚合图
        plot_combined_metrics(
            ea=event_accumulator_data, 
            tags=map_tags, 
            title="Validation mAP Curves", # 
            output_path=os.path.join(output_dir, "combined_mAPs.png")
        )
        
        # 绘制其他指标的聚合图 
        plot_combined_metrics(
            ea=event_accumulator_data,
            tags=other_tags,
            title="Learning Rate Curve", # 
            output_path=os.path.join(output_dir, "combined_learning_rate.png")
        )

        print("\n所有图表绘制完成！")