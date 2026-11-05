# Transformer 整体架构：编码器、解码器、输入输出

import torch
from torch import Tensor, nn

from .attention import make_decoder_mask, make_padding_mask
from .layers import DecoderBlock, EncoderBlock, PositionalEncoding, TokenEmbedding


"""
Encoder：嵌入 -> 位置编码 -> 多个编码器块

输入：
    embedding：词嵌入层 [B, S, d_model]
    num_layers：编码器块数量
"""
class Encoder(nn.Module):
    def __init__(self, embedding, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.input_layers = nn.Sequential(
            embedding,
            PositionalEncoding(d_model, dropout),
        )
        self.blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    """
    forward：
    输入：
        source_input_ids：源句 token ID，[B, S]
        source_mask：源句 padding 掩码，[B, 1, 1, S]
    返回：
        编码器最终特征，[B, S, d_model]
    """
    def forward(self, source_input_ids: Tensor, source_mask: Tensor | None = None) -> Tensor:
        X = self.input_layers(source_input_ids)
        for block in self.blocks:
            X = block(X, source_mask=source_mask)
        return X


"""
Decoder：嵌入 -> 位置编码 -> 多个解码器块

输入：
    embedding：词嵌入层 [B, S, d_model]
    num_layers：解码器块数量
"""
class Decoder(nn.Module):
    def __init__(self, embedding, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.embedding = embedding
        self.position = PositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    """
    forward：
    输入：
        target_input_ids：目标输入 ID，[B, T]
        encoder_output：编码器最终输出的特征，[B, S, d_model]
        source_mask、target_mask：源句/目标 掩码
        caches：各层的 K/V 缓存，训练时为 None，预测时为字典
    返回：
        解码器最终特征，[B, T, d_model]
    """
    def forward(
        self,
        target_input_ids: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor | None,
        target_mask: Tensor | None,
        caches
    ) -> Tensor:
        start_pos = 0  # 每次预测时的起始位置，训练时为 0
        if caches is not None:
            if "k" in caches[0]:  # 第一层的缓存字典中有 k，说明不是第一次预测
                start_pos = caches[0]["k"].shape[2]  # [B, 8, T, 64] 中的变量 T

        X = self.embedding(target_input_ids)  # [B, T, d_model]
        X = self.position(X, start_pos=start_pos)
        i = 0  # 用于记录 caches 的层数
        for block in self.blocks:
            cache = None if caches is None else caches[i]
            X = block(X, encoder_output, source_mask=source_mask, target_mask=target_mask, cache=cache)
            i += 1
        return X


"""
Transformer：组装编码器、解码器与输出投影

输入：
    vocab_size：共同词表大小
    d_model、num_heads、d_ff、dropout：从 configs/base.json 读取
"""
class Transformer(nn.Module):
    def __init__(self, vocab_size, pad_id, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.pad_id = pad_id
        embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.encoder = Encoder(embedding, num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(embedding, num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)
        self.output_layer.weight = embedding.embedding.weight  # 三处共用同一份参数  [V, d_model]

        # 普通 Linear 使用 Xavier 初始化
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.output_layer:  # 找出全连接层并排除输出层
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 这里使用了正态分布，N(0, d_model ** -0.5)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.output_layer.weight[pad_id].zero_()  # 再将输出层矩阵 <pad> 行置零

    # encode：源 ID -> 编码器最终特征、源句 padding 掩码
    def encode(self, source_input_ids: Tensor) -> tuple[Tensor, Tensor]:
        source_mask = make_padding_mask(source_input_ids, self.pad_id)
        encoder_output = self.encoder(source_input_ids, source_mask=source_mask)
        return encoder_output, source_mask

    def decode(self, target_input_ids: Tensor, encoder_output: Tensor, source_mask: Tensor | None, caches = None) -> Tensor:
        target_mask = make_decoder_mask(target_input_ids, self.pad_id, incremental=caches is not None)
        return self.decoder(target_input_ids, encoder_output, source_mask=source_mask, target_mask=target_mask, caches=caches)

    # init_cache：为每层单独创建一个空字典，形如 [{}, {}, ...]
    def init_cache(self) -> list[dict[str, Tensor]]:
        return [{} for _ in self.decoder.blocks]

    """
    forward：整段源句与目标输入 -> 词表分数
    输入：
        source_input_ids、target_input_ids：含有<pad>的 token ID
    返回：
        logits：原始分数   [B, T, V]
    """
    def forward(self, source_input_ids: Tensor, target_input_ids: Tensor) -> Tensor:
        encoder_output, source_mask = self.encode(source_input_ids)
        decoder_output = self.decode(target_input_ids, encoder_output, source_mask)
        logits = self.output_layer(decoder_output)
        return logits