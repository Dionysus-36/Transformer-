# 忽略 PAD 位置的标签平滑交叉熵

from torch import Tensor, nn


"""
compute_loss：计算有效目标位置的标签平滑交叉熵

输入：
    logits：DP 汇集到 GPU 0 的词表原始分数  [B, T, V]
    labels：训练标签                      [B, T]
    pad_id
返回：
    loss_sum：非 PAD 位置的损失总和
    num_tokens：非 PAD 标签的数量，包含 <eos>

标签平滑将独热标签分布与词表均匀分布按 0.9 和 0.1 混合。
返回总和与数量，供训练时按总 batch 的有效 token 数计算平均值。
"""
def compute_loss(logits: Tensor, labels: Tensor, pad_id) -> tuple[Tensor, Tensor]:
    # 原文采用标签平滑；先求损失总和，训练时再按有效 token 总数取平均
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1, reduction="sum")
    # 交叉熵要求类别维放在第二维，因此交换 T 与 V
    loss_sum = loss_fn(logits.transpose(1, 2), labels)
    num_tokens = (labels != pad_id).sum()
    return loss_sum, num_tokens