# 嵌入、正弦位置编码、FFN、Post-LN 与编码器/解码器块

import math

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention


"""
TokenEmbedding：根据 token ID 查表，并乘 sqrt(d_model)

输入：
    vocab_size：共同词表大小，由预处理结果提供
    d_model：词向量维数
    pad_id：<pad> 编号
"""
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # 原论文这么做的。我认为是为了让 embedding 的方差与位置编码的方差相当，控制在 N(0,1)，同时让输出层的 E.T 尽量小
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids) * self.scale  # [B, L, d_model]


"""
PositionalEncoding：加入位置信息，再做 dropout
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    """
    forward：按当前长度和起始位置生成位置编码
    输入：
        X：TokenEmbedding的结果，[B, L, d_model]
        start_pos：起始位置；训练时为 0，预测时为已有位置数
    返回：
        加入位置编码后的特征，与 X 形状相同
    """
    def forward(self, X: Tensor, start_pos = 0) -> Tensor:
        positions = torch.arange(
            start_pos,
            start_pos + X.shape[1],
            device=X.device,
        ).unsqueeze(1)  # [L, 1]

        dimensions = torch.arange(
            0,
            self.d_model,
            2,
            device=X.device,
        )  # [d_model / 2]，0-d_model，步长为 2

        angles = positions / torch.pow(10000.0, dimensions / self.d_model)  # 即 pos / 10000^(2i/d_model)

        encoding = torch.zeros(
            X.shape[1],
            self.d_model,
            device=X.device,
        )  # [L, d_model]，即 PE

        encoding[:, 0::2] = torch.sin(angles)  # 偶数列用 sin
        encoding[:, 1::2] = torch.cos(angles)  # 奇数列用 cos

        return self.dropout(X + encoding.unsqueeze(0))


"""
AddNorm：残差相加后做 LayerNorm

输入：
    d_model：特征维数
    dropout：子层输出的暂退概率
"""
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    # LayerNorm(X + Dropout(Y))
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return self.norm(X + self.dropout(Y))


"""
EncoderBlock：自注意力 -> Add & Norm -> FFN -> Add & Norm

输入：
    d_model：特征维数
    num_heads：注意力头数
    d_ff：FFN 中间层维数
    dropout：残差分支的暂退概率
输出：
    [B, S, d_model]
"""
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.addnorm2 = AddNorm(d_model, dropout)

    # source_mask：源句 padding 掩码，[B, 1, 1, S]
    def forward(self, X: Tensor, source_mask: Tensor | None = None) -> Tensor:
        Y = self.self_attention(X, X, X, mask=source_mask)
        X = self.addnorm1(X, Y)
        Y = self.ffn(X)
        return self.addnorm2(X, Y)


"""
DecoderBlock：自注意力 -> Add & Norm -> 交叉注意力 -> Add & Norm -> FFN -> Add & Norm
输出：
    [B, T, d_model]
"""
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.addnorm3 = AddNorm(d_model, dropout)

    """
    forward：结合目标前文和源句信息，更新目标特征
    输入：
        X：目标特征，预测时 T 为 1            [B, T, d_model]
        encoder_output：编码器最终输出        [B, S, d_model]
        source_mask：源句 padding 掩码       [B, 1, 1, S]
        target_mask：目标 padding 与因果掩码  [B, 1, T, T]
        cache：本层预测 K/V 缓存
    """
    def forward(
        self,
        X: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        cache: dict[str, Tensor] | None = None,
    ) -> Tensor:
        Y = self.self_attention(X, X, X, mask=target_mask, cache=cache)
        X = self.addnorm1(X, Y)

        # Q 来自目标侧，K/V 来自编码器最终输出
        Y = self.cross_attention(X, encoder_output, encoder_output, mask=source_mask)
        X = self.addnorm2(X, Y)
        Y = self.ffn(X)
        return self.addnorm3(X, Y)