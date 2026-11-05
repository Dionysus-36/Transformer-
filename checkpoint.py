# 训练进度保存、恢复与最后 5 份模型权重平均

import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam


"""
save_checkpoint：保存一份用于续训的完整检查点 last.pt；单独保存当前这一步的模型权重；删除过旧的权重文件，只保留最近 5 份

输入：
    run_dir：本次训练结果目录
    model：参与训练的模型，DP 包装后的模型也可传入
    optimizer：Adam 优化器
    progress：训练进度字典，由 train.py 提供
    形如：
        {"step": 1000, "epoch": 0, "start_batch": 4000, "elapsed_seconds": 3600.0}
        step：已完成的参数更新次数
        epoch：当前轮次，从 0 开始
        start_batch：本轮已消费的总 batch 数
        elapsed_seconds：累计训练耗时，单位为秒
    config：实际模型与训练配置，包括 warmup_steps、batch 大小及梯度累积设置
    metadata：数据预处理生成的字典，包含词表大小和特殊 token ID 等信息
返回：
    last_path：完整检查点 last.pt 的路径

在一次参数更新完成、梯度清空后保存，不保存尚未完成的梯度累积。
model.state_dict() 中各权重张量保持原有 shape。
"""
def save_checkpoint(run_dir, model: nn.Module, optimizer: Adam, progress, config, metadata) -> Path:
    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model = model.module if isinstance(model, nn.DataParallel) else model

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": progress,
        "config": config,
        "metadata": metadata,
        # 保存随机数生成器状态
        "random_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }

    # 先写临时文件，写完再替换，避免中途停止时破坏上一份完整检查点
    last_path = run_dir / "last.pt"
    temporary_path = run_dir / "last.tmp"
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(last_path)

    # 保存当前 step 的权重
    weights_path = weights_dir / f"step-{progress['step']:08d}.pt"
    temporary_path = weights_path.with_suffix(".tmp")
    torch.save({
        "model": checkpoint["model"],
        "step": progress["step"],
        "config": config,
        "metadata": metadata,
    }, temporary_path)
    temporary_path.replace(weights_path)

    for old_path in sorted(weights_dir.glob("step-*.pt"))[:-5]:  # 只保留最近 5 份权重
        old_path.unlink()
    return last_path


"""
load_checkpoint：恢复模型参数和 Adam 状态

输入：
    path：完整检查点 last.pt 的路径
    model：已放到训练设备上的模型，可在 DP 包装前恢复
    optimizer：已经创建好的 Adam 优化器
返回：
    checkpoint：读取的完整字典，训练程序从中取得进度、配置及数据说明

学习率调度没有独立计数器；下一次使用 progress["step"] + 1 计算学习率。
随机状态由 restore_random_state 在数据迭代器创建后恢复。
跨 Kaggle 会话续训时，训练入口还需将旧 run/weights/ 复制到新结果目录。
"""
def load_checkpoint(path, model: nn.Module, optimizer: Adam) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = model.module if isinstance(model, nn.DataParallel) else model
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer.zero_grad()
    model.train()  # 切换到训练模式，启用 dropout
    return checkpoint


"""
restore_random_state：恢复生成器状态

输入：
    checkpoint：load_checkpoint 返回的字典

先根据 progress 设置采样器的 epoch 和 start_batch，再创建 iter(train_iter)，
最后调用本函数，避免迭代器初始化消耗随机数，改变后续 dropout 的结果。
"""
def restore_random_state(checkpoint):
    random_state = checkpoint["random_state"]
    random.setstate(random_state["python"])
    torch.set_rng_state(random_state["torch"])
    torch.cuda.set_rng_state_all(random_state["cuda"])


"""
average_checkpoints：平均最近的模型权重，生成最终测试使用的 averaged.pt

输入：
    run_dir：包含 weights/ 的训练结果目录
    num_checkpoints：取最近多少份权重
返回：
    output_path：averaged.pt 的路径

逐项对相同位置的参数取算术平均，shape 保持不变。
不足 5 份时使用已有权重，并在输出的 steps 中记录实际采用的更新次数。
"""
def average_checkpoints(run_dir, num_checkpoints=5) -> Path:
    run_dir = Path(run_dir)
    paths = sorted((run_dir / "weights").glob("step-*.pt"))[-num_checkpoints:]
    if not paths:
        raise FileNotFoundError("weights/ 中没有可用于平均的模型权重。")

    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)

    averaged_weights = {name: weight.clone() for name, weight in checkpoint["model"].items()}
    steps = [checkpoint["step"]]  # 记录实际保存的步数
    for path in paths[1:]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for name in averaged_weights:
            averaged_weights[name].add_(checkpoint["model"][name])  # 原地累加
        steps.append(checkpoint["step"])

    for weight in averaged_weights.values():
        weight.div_(len(paths))  # 原地除以权重数量，得到平均值

    output_path = run_dir / "averaged.pt"
    temporary_path = run_dir / "averaged.tmp"
    torch.save({
        "model": averaged_weights,
        "config": checkpoint["config"],
        "metadata": checkpoint["metadata"],
        "steps": steps,
    }, temporary_path)
    temporary_path.replace(output_path)
    return output_path
