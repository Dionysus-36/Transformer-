# 使用 K/V 缓存逐词生成：束搜索与长度惩罚

import torch
from torch import Tensor, nn


"""
length_penalty：计算束搜索的长度惩罚

输入：
    length：已经生成的 token 数，不包含 <bos>，完成时包含 <eos>
    alpha：长度惩罚系数，最终测试使用 0.6
返回：
    惩罚因子，用累计对数概率除以它，得到候选译文的分数

采用论文引用的 GNMT 形式：((5 + length) / 6) ** alpha。
"""
def length_penalty(length, alpha=0.6) -> float:
    return ((5 + length) / 6) ** alpha


"""
beam_search：保留 4 个候选前缀，按长度惩罚后的分数选择译文。一次计算一个 batch

输入：
    model：放在预测设备上的原始 Transformer
    source_input_ids：已添加 <eos>、右侧补齐的英文 ID，[B, S]
    bos_id、eos_id
    beam_size
    alpha：非负长度惩罚系数，最终测试使用 0.6
返回：
    predictions：按输入顺序保存的最佳德文 ID 列表
        不包含 <bos>、<eos> 和 <pad>，后续由 evaluate.py 还原成文本

每句最多生成英文非 PAD 长度加 50 个 token，英文长度包含已有的 <eos>。
达到长度上限时优先使用已完成译文；尚未生成 <eos> 时返回最佳未完成前缀。
预测时关闭 dropout 和梯度记录。
"""
def beam_search(model: nn.Module, source_input_ids: Tensor, bos_id, eos_id, beam_size=4, alpha=0.6) -> list[list[int]]:
    model.eval()  # 关闭 dropout
    source_input_ids = source_input_ids.to(next(model.parameters()).device)  # 放在预测设备上
    limits = ((source_input_ids != model.pad_id).sum(dim=1) + 50).tolist()  # [B]，每行是生成长度上限
    predictions = []

    with torch.no_grad():
        encoder_output, source_mask = model.encode(source_input_ids)  # [B, S, d_model]、[B, 1, 1, S]
        for i in range(source_input_ids.shape[0]):
            sequences = source_input_ids.new_full((1, 1), bos_id)  # [K, L + 1]，K 为未完成候选数，L 不含 <bos>
            scores = encoder_output.new_zeros(1)  # [K]，累计对数概率，初始为 log(1)
            caches = model.init_cache()
            best_score = float("-inf")  # 最佳已完成译文的分数
            best_translation = None  # 尚未生成 <eos>

            for step in range(limits[i]):
                num_beams = sequences.shape[0]
                decoder_output = model.decode(
                    sequences[:, -1:],  # [K, 1]，只输入每条候选最后一个词
                    encoder_output[i:i + 1].expand(num_beams, -1, -1),  # 扩展为 [K, S, d_model]
                    source_mask[i:i + 1].expand(num_beams, -1, -1, -1),
                    caches=caches,
                )  # [K, 1, d_model]
                logits = model.output_layer(decoder_output[:, 0, :])  # [K, V]
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs[:, [model.pad_id, bos_id]] = float("-inf")  # 禁止生成 <pad> 和 <bos>
                candidate_scores = scores.unsqueeze(1) + log_probs  # [K, V]，每个前缀扩展整个词表
                vocab_size = logits.shape[-1]
                # 多取一些候选，给 <eos> 留出位置，继续保留最多 beam_size 个未完成前缀
                top_scores, top_indices = candidate_scores.flatten().topk(2 * beam_size)
                values = top_scores.tolist()
                indices = top_indices.tolist()
                parents = []
                tokens = []
                next_scores = []
                for j in range(len(indices)):
                    if values[j] == float("-inf"):
                        continue
                    parent = indices[j] // vocab_size
                    token = indices[j] % vocab_size
                    if token == eos_id:
                        score = values[j] / length_penalty(step + 1, alpha)
                        if score > best_score:
                            best_score = score
                            best_translation = sequences[parent, 1:].tolist()
                    elif len(parents) < beam_size:
                        parents.append(parent)
                        tokens.append(token)
                        next_scores.append(values[j])

                if not parents:
                    break
                parent_indices = source_input_ids.new_tensor(parents)  # [K]，可能重复，同一前缀可以扩展出多个候选
                next_ids = source_input_ids.new_tensor(tokens).unsqueeze(1)  # [K, 1]
                sequences = torch.cat((sequences.index_select(0, parent_indices), next_ids), dim=1)
                scores = encoder_output.new_tensor(next_scores)
                for cache in caches:
                    cache["k"] = cache["k"].index_select(0, parent_indices)  # [K, H, L, head_dim]
                    cache["v"] = cache["v"].index_select(0, parent_indices)

                # 后续对数概率只能减少；按最大允许长度计算最乐观的分数，判断能否提前结束
                best_possible = scores.max().item() / length_penalty(limits[i], alpha)
                if best_score >= best_possible:
                    break

            if best_translation is not None:
                predictions.append(best_translation)
            else:
                best = scores.argmax().item()
                predictions.append(sequences[best, 1:].tolist())  # 去掉 <bos>

    return predictions
