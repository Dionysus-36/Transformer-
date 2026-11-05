# 读取训练日志，绘制训练损失、验证损失和学习率曲线

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 直接保存图片，不打开图形窗口
import matplotlib.pyplot as plt


"""
plot：读取 train.csv，将三条曲线分别保存为 PNG

输入：
    run_dir：训练结果目录，里面有 train.csv
返回：
    out_dir：图片目录 run_dir/plots/

横轴均为参数更新次数 step；验证损失只使用执行验证时记录的值。
"""
def plot(run_dir) -> Path:
    run_dir = Path(run_dir)
    with (run_dir / "train.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 读取训练日志
    for name, label in (("train_loss", "Training loss"), ("eval_loss", "Validation loss"), ("lr", "Learning rate")):
        # 每个坐标组成列表
        points = [(int(row["step"]), float(row[name])) for row in rows if row[name] != ""]
        steps = [point[0] for point in points]
        values = [point[1] for point in points]

        fig, ax = plt.subplots(figsize=(7, 4))  # 创建画布和坐标轴
        ax.plot(steps, values, marker="o" if name == "eval_loss" else None)  # 绘制曲线
        ax.set(xlabel="Step", ylabel=label)  # 设置坐标轴标签
        ax.grid(alpha=0.3)  # 显示网格线
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close(fig)

    print(f"曲线已保存：{out_dir}", flush=True)
    return out_dir


"""
main：读取训练结果目录，启动绘图
"""
def main():
    parser = argparse.ArgumentParser(description="绘制 Transformer 训练曲线")
    parser.add_argument("--run-dir", type=Path, required=True)
    plot(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
