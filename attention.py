# 缩放点积注意力、多头注意力与掩码

import math

import torch
from torch import Tensor, nn


"""
make_padding_mask：找出 <pad> 的位置，转化为 True/False，作为 padding 掩码
输入：
    token_ids：token 编号组成的向量
    pad_id：<pad> 编号
返回：
    [B, 1, 1, Lk]
"""
def make_padding_mask(token_ids: Tensor, pad_id: int) -> Tensor:
    if token_ids.ndim != 2:
        raise ValueError("token_ids error")

    return token_ids.eq(pad_id).unsqueeze(1).unsqueeze(1)


"""
make_causal_mask：生成因果掩码,training 时使用
输入：
    seq_length：序列长度 T
返回：
    [1, 1, T, T]
"""
def make_causal_mask(seq_length: int, device) -> Tensor:
    if seq_length <= 0:
        raise ValueError("seq_length error")

    future_positions = torch.ones(
        seq_length,  # 对应 query 位置
        seq_length,  # 对应 key 位置
        dtype=torch.bool,
        device=device,
    ).triu(diagonal=1)  # [T, T]，上三角为 True，表示屏蔽未来位置

    return future_positions.unsqueeze(0).unsqueeze(0)


"""
make_decoder_mask：训练时合并两种掩码，预测时返回 None
输入：
    target_input_ids：训练时为 [B, T]
    pad_id：<pad> 编号
    incremental：是否使用之前的预测
返回：
    训练时为 [B, 1, T, T]，预测时为 None
"""
def make_decoder_mask(target_input_ids: Tensor, pad_id: int, incremental: bool = False) -> Tensor | None:
    if incremental:
        return None

    padding_mask = make_padding_mask(target_input_ids, pad_id)
    causal_mask = make_causal_mask(
        target_input_ids.shape[1],  # [B, T] -> T
        device=target_input_ids.device,
    )

    # 两种掩码中任一为 True，就屏蔽该位置
    return padding_mask | causal_mask  # 广播为 [B, 1, T, T]



"""
缩放点积注意力

B: batch_size
H: num_heads
Lq: query 序列长度
Lk: key/value 序列长度

输入：
    queries：分头后的 query  [B, H, Lq, dk]
    keys：分头后的 key       [B, H, Lk, dk]
    values：分头后的 value   [B, H, Lk, dv]
    mask：布尔掩码，广播到    [B, H, Lq, Lk]，True 表示屏蔽
返回：
    output：加权求和结果 [B, H, Lq, dv]
    weights：注意力权重  [B, H, Lq, Lk]
"""
class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        dk = queries.shape[-1]

        scores = torch.matmul(
            queries / math.sqrt(dk),
            keys.transpose(-2, -1),  # [B, H, Lk, dk] -> [B, H, dk, Lk]
        )

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))  # 将 True 的位置置为 负无穷，softmax 后为 0

        weights = torch.softmax(scores, dim=-1)  # 沿着 key 维度

        # 加权求和
        output = torch.matmul(weights, values)
        return output, weights


"""
多头注意力：包括 Q/K/V 投影、分头、合头和输出投影

输入：
    d_model：输入和输出的向量维数
    num_heads：注意力头数
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention()
        self.attention_weights = None  # 把注意力权重保存一下

    """
    _split_heads：分头
    输入：
        projected：投影后的向量，[B, L, d_model]
    返回：
        [B, H, L, head_dim]
    """
    def _split_heads(self, projected: Tensor) -> Tensor:
        batch_size, length, _ = projected.shape
        return projected.reshape(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    """
    _merge_heads：拼接
    输入：
        head_outputs：各头输出，[B, H, Lq, head_dim]
    返回：
        [B, Lq, d_model]
    """
    def _merge_heads(self, head_outputs: Tensor) -> Tensor:
        batch_size, _, query_length, _ = head_outputs.shape

        return head_outputs.transpose(1, 2).contiguous().view(batch_size, query_length, self.d_model)

    """
    输入：
        cache：预测自注意力的 K/V 缓存，首次传空字典，每层单独保存
    返回：
        [B, Lq, d_model]
    """
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor | None = None,
        cache: dict[str, Tensor] | None = None,
    ) -> Tensor:
        q_heads = self._split_heads(self.W_q(queries))
        k_heads = self._split_heads(self.W_k(keys))
        v_heads = self._split_heads(self.W_v(values))

        if cache is not None and "k" in cache:
            k_heads = torch.cat((cache["k"], k_heads), dim=2)
            v_heads = torch.cat((cache["v"], v_heads), dim=2)

        head_outputs, weights = self.attention(q_heads, k_heads, v_heads, mask)
        if cache is not None:
            cache["k"] = k_heads.detach()
            cache["v"] = v_heads.detach()  # detach() 避免影响反向传播

        self.attention_weights = weights.detach()

        merged = self._merge_heads(head_outputs)
        return self.W_o(merged)