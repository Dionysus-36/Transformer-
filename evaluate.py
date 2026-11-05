# 最终测试：平均模型权重、束搜索翻译、还原文本与计算 BLEU-4

import argparse
import json
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from sacremoses import MosesDetokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import average_checkpoints
from .data import ParallelDataset
from .decode import beam_search
from .model import Transformer


"""
ids_to_text：把生成的 ID 还原为正常德文句子

输入：
    ids：一条译文的 ID 列表，不含 <bos>、<eos>、<pad>
    tokens：vocab.json 中按 ID 排列的 token 列表
    detokenizer：德文 Moses 反分词器
    separator：BPE 子词标记，沿用预处理设置
返回：
    合并子词、还原标点和 XML 转义字符后的德文文本

例如：Stud@@ ent . -> Student.
"""
def ids_to_text(ids: list[int], tokens: list[str], detokenizer, separator) -> str:
    text = " ".join(tokens[token_id] for token_id in ids)  # 用空格连接 token
    # text = text.replace("@@ ", "").removesuffix("@@")
    text = text.replace(separator + " ", "").removesuffix(separator)
    return detokenizer.detokenize(text.split())


"""
evaluate：使用平均权重翻译完整 test，计算语料级 BLEU-4 并保存结果

输入：
    args：main 读取的数据目录、训练结果目录、输出目录和权重平均数量等参数
返回：
    result：BLEU 分数、评分 signature、句子数量和实际使用的检查点等信息

只读取测试集，保持原始顺序；参考译文直接读取预处理保存的原文。
"""
def evaluate(args) -> dict:
    test_dataset = ParallelDataset(args.data_dir, "test")
    metadata = test_dataset.metadata
    reference_path = args.data_dir / metadata["processing"]["reference_files"]["test"]
    # 把读到的参考译文按行分割成列表
    references = reference_path.read_text(encoding="utf-8").splitlines()
    if len(references) != len(test_dataset):
        raise ValueError("测试集与参考译文的句子数量不一致。")

    # 1. 先取模型参数
    checkpoint_path = average_checkpoints(args.run_dir, args.average_last)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    steps = checkpoint["steps"]  # 实际使用的检查点步数
    model = Transformer(metadata["vocab_size"], metadata["pad_id"], **checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])  # 把平均后的权重加载到模型
    del checkpoint
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. 再取词表、BPE 分隔符和反分词器
    vocab_path = args.data_dir / metadata["tokenizer"]["vocab"]
    tokens = json.loads(vocab_path.read_text(encoding="utf-8"))["tokens"]  # 词表
    separator = metadata["data_config"]["bpe_separator"]  # 分隔符
    detokenizer = MosesDetokenizer(lang=metadata["target_language"])  # 反分词器
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_test)

    translations = []
    for source_input_ids in tqdm(test_iter, desc="测试集翻译"):  # 显示进度条
        predictions = beam_search(model, source_input_ids, metadata["bos_id"], metadata["eos_id"])
        for ids in predictions:
            translations.append(ids_to_text(ids, tokens, detokenizer, separator))

    # 整个测试集一起计算，保留大小写，默认使用 1～4 元 BLEU
    bleu = BLEU(tokenize="13a")
    score = bleu.corpus_score(translations, [references])
    result = {
        "bleu": score.score,
        "signature": str(bleu.get_signature()),
        "num_sentences": len(translations),
        "checkpoint_steps": steps,
        "beam_size": 4,
        "alpha": 0.6,
        "test_set": metadata["processing"]["test_set"],
    }
    out_dir = args.out_dir or args.run_dir / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "translations.de").write_text("\n".join(translations) + "\n", encoding="utf-8")
    (out_dir / "bleu.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(score, flush=True)
    print(f"结果已保存：{out_dir}", flush=True)
    return result


"""
main：读取命令行参数，启动最终测试

默认平均最后 5 份模型权重，测试 batch_size 为 32，结果保存到 run_dir/test/。
不足 5 份时使用已有权重，实际检查点步数记录在 bleu.json 中。
"""
def main():
    parser = argparse.ArgumentParser(description="Transformer Base 测试集翻译与 BLEU-4")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--average-last", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
