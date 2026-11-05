# 读取预处理分片、按长度组 batch、padding 与解码器输入/标签构造
"""
德文句子需要进行特殊处理。分为训练标签和解码器输入两种。例如：
解码器输入：   <bos>   Ich     renne
训练标签：      Ich   renne    <eos>
"""

import json
import random
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


"""
ParallelDataset：读取一个划分的英德句子 ID，尚未补齐

输入：
    data_dir：预处理结果目录，里面有 metadata.json、train/、eval/、test/
    split：train、eval 或 test

metadata.json 下形如：
{
    "splits": {
        "train": {
            "shards": [
                {"path": "train/shard-00000", "num_pairs": 25000},
                {"path": "train/shard-00001", "num_pairs": 25000}
            ]
        },
        "eval": {
            "shards": [
                {"path": "eval/shard-00000", "num_pairs": 3000}
            ]
        },
        "test": {
            "shards": [
                {"path": "test/shard-00000", "num_pairs": 3003}
            ]
        }
    }
}

"""
class ParallelDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = Path(data_dir)
        self.metadata = json.loads((self.data_dir / "metadata.json").read_text(encoding="utf-8"))  # 把metadata.json读取成字典
        self.shards = self.metadata["splits"][split]["shards"]  # 获取当前划分的分片列表

        # 记录每个分片在整个划分中的起止编号
        self.shard_offsets = [0]
        for shard in self.shards:
            self.shard_offsets.append(self.shard_offsets[-1] + shard["num_pairs"])
        self.shard_offsets = torch.tensor(self.shard_offsets)  # [K + 1]，K 为分片数

        self.current_shard = -1  # 尚未打开任何分片
        self.arrays = {}  # 只保存当前分片的 ID 和偏移数组

    """
    __len__：返回当前划分的句子对总数
    """
    def __len__(self):
        return self.shard_offsets[-1].item()

    """
    __getitem__：取得 index 对应的句子对
    返回：
        source_ids：当前英文句子，包含 <eos>，[S_i]
        target_ids：当前德文句子，不含起止标记，[L_i]

    注意：shard_offsets 用于定位句子所在的分片，
         而arrays["source_offsets"] 和 arrays["target_offsets"] 用于定位句子在某个分片内的偏移
    """
    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        # .searchsorted() 用于在有序数组中查找插入位置，right=True 表示返回第一个大于 index 的位置

        shard_index = torch.searchsorted(self.shard_offsets, index, right=True).item() - 1  # 当前句子所在分片的全局索引
        local_index = index - self.shard_offsets[shard_index].item()  # 当前句子在分片内的局部索引
        # 切换分片时才打开文件，同一分片内的样本复用这些数组
        if shard_index != self.current_shard:
            shard_dir = self.data_dir / self.shards[shard_index]["path"]
            arrays = {}
            for name in ("source_ids", "source_offsets", "target_ids", "target_offsets"):
                arrays[name] = np.load(shard_dir / f"{name}.npy", mmap_mode="r")
            self.arrays = arrays
            self.current_shard = shard_index
        arrays = self.arrays

        source_offsets = arrays["source_offsets"]
        target_offsets = arrays["target_offsets"]

        # 找出当前句子的 ID 列表，直接创建 long 张量，避免先复制为 int32 再转换
        source_ids = torch.tensor(arrays["source_ids"][source_offsets[local_index]:source_offsets[local_index + 1]], dtype=torch.long)
        target_ids = torch.tensor(arrays["target_ids"][target_offsets[local_index]:target_offsets[local_index + 1]], dtype=torch.long)
        return source_ids, target_ids

    """
    collate_train：为 train、eval 构造解码器输入和标签，再分别在右侧补齐

    输入：
        batch：一批尚未补齐的英德 ID 对
    返回：
        source_input_ids：英文输入，[B, S]
        target_input_ids：德文输入，开头添加 <bos>，[B, T]
        labels：德文标签，末尾添加 <eos>，与 target_input_ids 形状相同

    B 为两张 GPU 共同处理的总句子对数，S 为最长源句长度，T 为最长德文长度加 1。
    先补齐整个 batch，再由 DP 拆分，保证本次两张卡的序列长度一致。
    """
    def collate_train(self, batch: list) -> tuple[Tensor, Tensor, Tensor]:
        pad_id = self.metadata["pad_id"]
        bos_id = self.metadata["bos_id"]
        eos_id = self.metadata["eos_id"]
        source_sequences = []
        target_inputs = []
        target_labels = []
        for source_ids, target_ids in batch:
            source_sequences.append(source_ids)
            target_inputs.append(torch.cat((target_ids.new_tensor([bos_id]), target_ids)))
            target_labels.append(torch.cat((target_ids, target_ids.new_tensor([eos_id]))))

        # batch_first=True 是将 batch 维度放在最前面
        source_input_ids = pad_sequence(source_sequences, batch_first=True, padding_value=pad_id)
        target_input_ids = pad_sequence(target_inputs, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(target_labels, batch_first=True, padding_value=pad_id)
        return source_input_ids, target_input_ids, labels

    """
    collate_test：将英文 ID 补齐，返回供束搜索使用的英文输入
    """
    def collate_test(self, batch: list) -> Tensor:
        source_sequences = [source_ids for source_ids, _ in batch]
        return pad_sequence(source_sequences, batch_first=True, padding_value=self.metadata["pad_id"])


"""
LengthBatchSampler：按长度组成双 GPU DP 处理的总 batch

输入：
    dataset：训练集的 ParallelDataset
    batch_size：两张 GPU 共同处理的总句子对数量
    bucket_size：每次按长度排序的候选句子数量
    seed：随机种子，决定每个 epoch 的随机顺序

"""
class LengthBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size=4096, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0
        # 只为训练分组读取长度；验证和测试按顺序读取，不需要长度数组
        self.lengths = np.empty(len(dataset), dtype=np.int32)  # [N]，记录每个句子对的最大长度
        self.num_batches = 0
        start = 0
        for shard in dataset.shards:
            shard_dir = dataset.data_dir / shard["path"]
            source_lengths = np.load(shard_dir / "source_lengths.npy", mmap_mode="r")
            target_lengths = np.load(shard_dir / "target_lengths.npy", mmap_mode="r")
            end = start + shard["num_pairs"]
            # 源句已有 <eos>，德文输入和标签都比保存的目标句多一个位置
            self.lengths[start:end] = np.maximum(source_lengths, target_lengths + 1)
            self.num_batches += math.ceil(shard["num_pairs"] / batch_size)
            start = end

    """
    set_epoch：用于续训
    输入：
        epoch：轮次，从 0 开始；每轮由训练程序更新
        start_batch：本轮已消费的总 batch 数，用于续训
    """
    def set_epoch(self, epoch, start_batch=0):
        self.epoch = epoch
        self.start_batch = start_batch

    """
    len：返回还剩多少 batch 没训练
    """
    def __len__(self):
        return max(0, self.num_batches - self.start_batch)

    def __iter__(self):
        # 用随机数生成器生成 shard 顺序
        rng = random.Random(self.seed + self.epoch)
        shard_order = list(range(len(self.dataset.shards)))
        rng.shuffle(shard_order)
        batch_index = 0

        for shard_index in shard_order:
            # 随机生成一个 shard 中的句子顺序
            start = self.dataset.shard_offsets[shard_index].item()
            end = self.dataset.shard_offsets[shard_index + 1].item()
            indices = list(range(start, end))
            rng.shuffle(indices)

            # 随机生成一个 bucket 内顺序，并按长度排序，确保一个 batch 内长度相近
            for position in range(0, len(indices), self.bucket_size):
                bucket = indices[position:position + self.bucket_size]
                indices[position:position + self.bucket_size] = sorted(bucket, key=lambda index: self.lengths[index])
            # 最先每隔 batch_size 切割，然后打乱顺序
            batches = [indices[position:position + self.batch_size] for position in range(0, len(indices), self.batch_size)]
            rng.shuffle(batches)
            for batch in batches:
                # 跳过续训前已经训练过的 batch
                if batch_index >= self.start_batch:
                    yield batch
                batch_index += 1


"""
load_data：创建训练、验证和测试的 DataLoader

输入：
    data_dir：Kaggle 挂载的 wmt14 目录
    batch_size：两张 GPU 共同处理的总句子对数量
    bucket_size、seed
    pin_memory：使用 GPU 时可开启，将 batch 放入便于向 GPU 传输的主机内存
返回：
    train_iter：每次提供 source_input_ids、target_input_ids、labels
    eval_iter：与训练集提供相同内容，保持验证顺序
    test_iter：每次仅提供英文输入，保持测试顺序
    metadata：数据说明

每轮训练前调用 train_iter.batch_sampler.set_epoch(epoch)。
由单个训练进程读取总 batch，补齐后交给 nn.DataParallel 拆分到两张 GPU。
续训时再传 start_batch；训练程序需要记录本轮实际消费了多少个 batch。
"""
def load_data(data_dir, batch_size, bucket_size=4096, seed=42, pin_memory=False) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    # 分别读取三个划分，共用同一词表和特殊 ID
    train_dataset = ParallelDataset(data_dir, "train")
    eval_dataset = ParallelDataset(data_dir, "eval")
    test_dataset = ParallelDataset(data_dir, "test")
    metadata = train_dataset.metadata
    batch_sampler = LengthBatchSampler(train_dataset, batch_size, bucket_size, seed)

    train_iter = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=train_dataset.collate_train,
        pin_memory=pin_memory,
    )
    eval_iter = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_train,
        pin_memory=pin_memory,
    )
    test_iter = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate_test,
        pin_memory=pin_memory,
    )
    return train_iter, eval_iter, test_iter, metadata
