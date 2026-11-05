# 单进程双 GPU DP 训练、验证与续训

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .checkpoint import load_checkpoint, restore_random_state, save_checkpoint
from .data import load_data
from .loss import compute_loss
from .model import Transformer
from .optim import create_optimizer, update_learning_rate


"""
compute_eval_loss：计算整个 eval 的平均损失

输入：
    model：当前训练模型，可使用 nn.DataParallel 包装
    eval_iter：每批提供英文 ID [B, S]、德文输入 ID 与标签 [B, T]
    pad_id：<pad> 编号
    device：输入和标签所在设备，DP 训练时为 cuda:0
返回：
    eval_loss：损失总和除以有效目标 token 总数，浮点数

使用与训练相同的标签平滑损失；验证时关闭 dropout，不计算梯度。
仅用于训练过程中的验证，完成后切回训练模式，不更新参数或 step。
"""
def compute_eval_loss(model: nn.Module, eval_iter, pad_id, device) -> float:
    model.eval()  # 关闭 dropout
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():  # 不记录梯度
        for source_input_ids, target_input_ids, labels in eval_iter:
            source_input_ids = source_input_ids.to(device)
            target_input_ids = target_input_ids.to(device)  # 这里还是输入正确的结果
            labels = labels.to(device)
            logits = model(source_input_ids, target_input_ids)  # [B, T, V]
            loss_sum, num_tokens = compute_loss(logits, labels, pad_id)
            total_loss += loss_sum.item()
            total_tokens += num_tokens.item()
            del logits, loss_sum, num_tokens

    model.train()
    return total_loss / total_tokens


"""
train：读取配置并完成一段训练，可从 last.pt 继续

输入：
    args：main 读取的配置路径、数据目录、结果目录、目标步数和续训路径
返回：
    progress：已完成的参数更新次数、数据读取位置、耗时与峰值显存

一次前向处理一个总 batch，由 DP 拆分到两张 GPU。
多个总 batch 累积梯度后，按有效目标 token 总数归一化，再更新一次参数。
只有完成参数更新后才保存检查点；max_steps 表示累计目标步数。
"""
def train(args) -> dict:
    if torch.cuda.device_count() != 2:
        raise RuntimeError("不是双 GPU 环境")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    max_steps = training["max_steps"]
    # 同时设置 PyTorch 的 CPU 和 GPU 种子，采样器独立使用 seed + epoch
    torch.manual_seed(training["seed"])
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None and ((run_dir / "last.pt").exists() or (run_dir / "train.csv").exists()):
        raise FileExistsError("结果目录已有训练记录，请使用 --resume 续训或更换结果目录。")

    train_iter, eval_iter, _, metadata = load_data(
        args.data_dir,
        training["batch_size"],
        bucket_size=training["bucket_size"],
        seed=training["seed"],
        pin_memory=True,
    )
    pad_id = metadata["pad_id"]
    device = "cuda:0"
    model = Transformer(metadata["vocab_size"], pad_id, **config["model"])
    model = nn.DataParallel(model.to(device), device_ids=[0, 1], output_device=0)
    optimizer = create_optimizer(model)

    if args.resume is None:
        progress = {
            "step": 0,
            "epoch": 0,
            "start_batch": 0,
            "elapsed_seconds": 0.0,
            "eval_seconds": 0.0,
            "eval_loss": None,
            "last_eval_step": -1,
            "peak_allocated_gib": [0.0, 0.0],
            "peak_reserved_gib": [0.0, 0.0],
        }
    else:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        progress = checkpoint["progress"]
        # 新 Kaggle 会话还需带上之前的权重和日志
        previous_dir = args.resume.parent
        if previous_dir.resolve() != run_dir.resolve():
            shutil.copytree(previous_dir / "weights", run_dir / "weights", dirs_exist_ok=True)
            shutil.copy2(previous_dir / "train.csv", run_dir / "train.csv")

    sampler = train_iter.batch_sampler
    sampler.set_epoch(progress["epoch"], progress["start_batch"])
    batches = iter(train_iter)
    if args.resume is not None:
        restore_random_state(checkpoint)
        del checkpoint

    # 保留已保存 step 以内的日志，避免续训时重复记录后面的行
    log_path = run_dir / "train.csv"
    previous_rows = []
    if log_path.exists():
        with log_path.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # 跳过列名
            previous_rows = [row for row in reader if int(row[0]) <= progress["step"]]
    columns = [
        "step", "epoch", "start_batch", "train_loss", "eval_loss", "lr",
        "source_tokens", "target_tokens", "step_seconds", "tokens_per_second",
        "elapsed_seconds", "eval_seconds", "gpu0_allocated_gib", "gpu1_allocated_gib",
        "gpu0_reserved_gib", "gpu1_reserved_gib",
    ]
    gpu_names = [torch.cuda.get_device_name(gpu) for gpu in range(2)]
    summary = {
        "precision": "FP32",
        "gpu_names": gpu_names,
        "torch_version": str(torch.__version__),
        "config": config,
        "progress": progress,
    }
    print(f"设备：{gpu_names}；精度：FP32；总 batch：{training['batch_size']}；累积次数：{training['accumulation_steps']}", flush=True)
    session_start = time.perf_counter()
    last_save = session_start

    with log_path.open("w", encoding="utf-8", newline="") as file, tqdm(
        total=max_steps, initial=progress["step"], desc="训练", unit="step"
    ) as bar:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(previous_rows)
        file.flush()
        for step in range(progress["step"] + 1, max_steps + 1):
            for gpu in range(2):
                torch.cuda.synchronize(gpu)
                torch.cuda.reset_peak_memory_stats(gpu)
            step_start = time.perf_counter()
            total_loss = 0.0
            source_tokens = 0
            target_tokens = 0

            for _ in range(training["accumulation_steps"]):
                try:
                    source_input_ids, target_input_ids, labels = next(batches)
                except StopIteration:
                    progress["epoch"] += 1
                    progress["start_batch"] = 0
                    sampler.set_epoch(progress["epoch"])
                    batches = iter(train_iter)
                    source_input_ids, target_input_ids, labels = next(batches)

                source_input_ids = source_input_ids.to(device, non_blocking=True)  # [B, S]
                target_input_ids = target_input_ids.to(device, non_blocking=True)  # [B, T]
                labels = labels.to(device, non_blocking=True)  # [B, T]
                logits = model(source_input_ids, target_input_ids)  # [B, T, V]
                loss_sum, num_tokens = compute_loss(logits, labels, pad_id)
                total_loss += loss_sum.item()
                source_tokens += (source_input_ids != pad_id).sum().item()
                target_tokens += num_tokens.item()
                loss_sum.backward()
                progress["start_batch"] += 1
                del logits, loss_sum

            # 累积的是损失总和的梯度，统一除以这次更新的有效目标 token 数
            for parameter in model.parameters():
                parameter.grad.div_(target_tokens)
            lr = update_learning_rate(optimizer, step, config["model"]["d_model"], training["warmup_steps"])
            optimizer.step()  # 更新参数
            optimizer.zero_grad()
            progress["step"] = step

            for gpu in range(2):
                torch.cuda.synchronize(gpu)
            step_seconds = time.perf_counter() - step_start
            progress["elapsed_seconds"] += step_seconds
            allocated = [torch.cuda.max_memory_allocated(gpu) / 1024 ** 3 for gpu in range(2)]
            reserved = [torch.cuda.max_memory_reserved(gpu) / 1024 ** 3 for gpu in range(2)]
            for gpu in range(2):
                progress["peak_allocated_gib"][gpu] = max(progress["peak_allocated_gib"][gpu], allocated[gpu])
                progress["peak_reserved_gib"][gpu] = max(progress["peak_reserved_gib"][gpu], reserved[gpu])

            finished = step == max_steps or (args.max_seconds is not None and time.perf_counter() - session_start >= args.max_seconds)
            eval_loss = ""
            # 准备进行一次验证
            if step % training["eval_every"] == 0 or finished:
                eval_start = time.perf_counter()
                eval_loss = compute_eval_loss(model, eval_iter, pad_id, device)
                progress["eval_seconds"] += time.perf_counter() - eval_start
                progress["eval_loss"] = eval_loss
                progress["last_eval_step"] = step

            train_loss = total_loss / target_tokens
            tokens_per_second = (source_tokens + target_tokens) / step_seconds
            writer.writerow([
                step, progress["epoch"], progress["start_batch"], train_loss, eval_loss, lr,
                source_tokens, target_tokens, step_seconds, tokens_per_second,
                progress["elapsed_seconds"], progress["eval_seconds"], *allocated, *reserved,
            ])
            file.flush()
            bar.update()
            if step % training["log_every"] == 0 or eval_loss != "" or finished:
                tqdm.write(f"step {step}：loss={train_loss:.4f}；eval={progress['eval_loss']}；lr={lr:.3g}；tokens/s={tokens_per_second:.0f}；GPU 峰值={allocated} GiB")

            if finished or time.perf_counter() - last_save >= training["save_every_seconds"]:
                save_checkpoint(run_dir, model, optimizer, progress, config, metadata)
                (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                last_save = time.perf_counter()
                tqdm.write(f"已保存：{run_dir / 'last.pt'}")
            if finished:
                break
    return progress


"""
main：读取命令行参数，启动训练

max_steps 为累计目标步数；max_seconds 可限制本场训练时长，
达到时会在当前参数更新完成后验证、保存并退出，需另外预留验证和文件保存时间。
"""
def main():
    parser = argparse.ArgumentParser(description="Transformer Base 双 GPU DP 训练")
    parser.add_argument("--config", type=Path, default=Path("configs/base.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-seconds", type=float)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
