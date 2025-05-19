import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_training_history(history_path, save_dir=None):
    # 读取历史数据
    with open(history_path, "r") as f:
        history = json.load(f)
    
    # 创建图表
    plt.figure(figsize=(12, 8))

    # 子图 1：损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.grid(True)
    plt.legend()

    # 子图 2：验证集加权准确率
    plt.subplot(2, 2, 2)
    plt.plot(history["epoch"], history["val_weighted_acc"], 'g-', label='Weighted Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Weighted Accuracy vs Epoch')
    plt.grid(True)
    plt.legend()

    # 子图 3：验证集非加权准确率
    plt.subplot(2, 2, 3)
    plt.plot(history["epoch"], history["val_unweighted_acc"], 'r-', label='Unweighted Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Unweighted Accuracy vs Epoch')
    plt.grid(True)
    plt.legend()

    # 子图 4：验证集加权 F1 值
    plt.subplot(2, 2, 4)
    plt.plot(history["epoch"], history["val_weighted_f1"], 'm-', label='Weighted F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation Weighted F1 vs Epoch')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    # 保存或显示图表
    if save_dir:
        save_path = Path(save_dir) / "training_history.png"
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化训练历史数据")
    parser.add_argument("--history_path", type=str, required=True, help="训练历史数据 JSON 文件路径")
    parser.add_argument("--save_dir", type=str, default=None, help="保存图表的目录（可选）")
    args = parser.parse_args()

    plot_training_history(args.history_path, args.save_dir)