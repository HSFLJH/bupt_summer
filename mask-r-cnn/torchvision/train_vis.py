import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_scalars(log_dir, tags, title="训练指标变化", output_path="metrics_plot.png"):
    """
    读取 TensorBoard 日志文件，提取指定指标，并绘制曲线。

    参数:
        log_dir: TensorBoard 日志目录（含 .tfevents 文件）
        tags: 要提取的 scalar 名称列表（如 'Loss/Total', 'Val/mask_mAP' 等）
        title: 图标题
        output_path: 保存图像路径
    """
    # 查找最新的事件文件
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        print(" 未找到 TensorBoard 事件日志文件")
        return
    event_path = os.path.join(log_dir, sorted(event_files)[-1])
    
    # 加载事件数据
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    
    # 绘制每个 tag 对应的曲线
    plt.figure(figsize=(10, 6))
    for tag in tags:
        if tag not in ea.Tags().get("scalars", []):
            print(f" 指标 {tag} 不存在于日志中")
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, label=tag)
    
    plt.title(title)
    plt.xlabel("Epoch / Step")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ 图像已保存为 {output_path}")


if __name__ == "__main__":
    log_dir = "/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/output/tensorboard"
    tags = ["Loss/Total", "Loss/Classifier", "Loss/Box", "Loss/Mask", "LR"]
    plot_tensorboard_scalars(log_dir, tags, title="训练过程指标变化", output_path="metrics_plot.png")
