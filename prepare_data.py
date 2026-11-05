# WMT14 英德语料预处理：下载、分词、联合 BPE、共同词表与 ID 分片
# 注意此处：英文末尾直接添加 <eos>，而德文暂不做处理

import argparse
import html
import json
import re
import shutil
import tarfile
from collections import Counter
from itertools import chain
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe


"""
main：读取配置和命令行参数，显式运行脚本时才启动下载与处理
"""
def main():
    parser = argparse.ArgumentParser(description="准备完整 WMT14 英德语料")
    parser.add_argument("--config", type=Path, default=Path("configs/base.json"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prepare_data(config["data"], args.work_dir, args.out)


"""
prepare_data：准备 train、eval、test；已有旧数据时只补 eval
输入：
    data_config：configs/base.json 中的 data 配置
    work_dir：原始包、原文和中间文本目录，Kaggle 中使用 /tmp/wmt14
    output_dir：最终数据目录，Kaggle 中使用 /kaggle/working/wmt14
返回：
    metadata：供后续训练与测试读取的数据说明
"""
def prepare_data(data_config: dict, work_dir: Path, output_dir: Path) -> dict:
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["data_config"] == data_config and "eval" in metadata["splits"]:
            print(f"数据已准备完成，可直接复用：{output_dir}", flush=True)
            return metadata

        # 只允许给旧数据增加 eval，训练、测试来源与预处理设置必须保持一致
        original_config = {
            **data_config,
            "sources": [source for source in data_config["sources"] if source["split"] != "eval"],
        }
        if metadata["data_config"] != original_config:
            raise ValueError("输出目录已有其他配置的数据，请使用新的输出目录。")
        return prepare_eval(data_config, work_dir, output_dir, metadata)

    work_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下载语料，取出配置中指定的英德文件
    corpora = {"train": [], "eval": [], "test": []}
    for source in data_config["sources"]:
        archive_path = work_dir / "downloads" / source["archive"]
        download_archive(source["url"], archive_path)
        raw_paths = extract_pair(archive_path, source["members"], work_dir / "raw" / source["name"])
        corpora[source["split"]].append({**source, "raw_paths": raw_paths})

    # 2. 清理和分词，保留 eval、test 原文
    tokenized_dir = work_dir / "tokenized"
    tokenization = {}
    for split in ("train", "eval", "test"):
        tokenization[split] = tokenize_split(split, corpora[split], tokenized_dir, output_dir)

    # 3. 只用训练集学习共同 BPE，再应用到三个划分
    codes_path = tokenizer_dir / "bpe.codes"
    learn_joint_bpe(tokenized_dir, codes_path, data_config)
    bpe_dir = work_dir / "bpe"
    apply_joint_bpe(tokenized_dir, bpe_dir, codes_path, data_config["bpe_separator"])
    # 4. 建立共同词表，将句子转成 ID 并分片保存
    token_to_id = build_vocab(bpe_dir, data_config, tokenizer_dir)
    splits = {}
    for split in ("train", "eval", "test"):
        splits[split] = save_split(split, bpe_dir, output_dir, token_to_id, data_config)

    # 5. 保存后续读取数据需要的信息
    metadata = {
        "format_version": 1,
        "dataset": "WMT14 English-German",
        "source_language": "en",
        "target_language": "de",
        "vocab_size": len(token_to_id),
        "pad_id": token_to_id["<pad>"],
        "unk_id": token_to_id["<unk>"],
        "bos_id": token_to_id["<bos>"],
        "eos_id": token_to_id["<eos>"],
        "tokenizer": {"vocab": "tokenizer/vocab.json", "bpe_codes": "tokenizer/bpe.codes"},
        "sequence_format": {
            "source": "BPE token IDs followed by <eos>",
            "target": "BPE token IDs only; data.py adds <bos>/<eos> for input/labels",
            "padding": "right padding by data.py, not stored in shards",
            "ids_dtype": "int32",
            "offsets_dtype": "int64",
            "lengths_dtype": "int32",
        },
        "processing": {
            "preserve_case": True,
            "tokenizer": "sacremoses; punctuation normalization, aggressive dash splitting, XML escaping",
            "bpe_training_split": "train",
            "vocab_training_split": "train",
            "length_filter_unit": "BPE tokens before source <eos>; train only",
            "eval_set": "newstest2013",
            "test_set": "newstest2014/full (SacreBLEU: wmt14/full)",
            "reference_files": {"eval": "eval/reference.de", "test": "test/reference.de"},
        },
        "data_config": data_config,
        "tokenization": tokenization,
        "splits": splits,
    }
    # 最后写入 metadata，避免未完成的数据被当作可直接训练的结果
    save_json(metadata_path, metadata)
    print(f"数据准备完成：{output_dir}", flush=True)
    return metadata


"""
prepare_eval：给已保存的 train、test 数据补充 newstest2013 验证集
输入：
    data_config、work_dir、output_dir：与 prepare_data 相同
    metadata：旧数据说明，BPE、词表和 train、test 分片已存在
返回：
    加入 eval 信息后的 metadata

复用已有 BPE 和词表，不重新下载或处理训练集、测试集。
"""
def prepare_eval(data_config: dict, work_dir: Path, output_dir: Path, metadata: dict) -> dict:
    source = next(source for source in data_config["sources"] if source["split"] == "eval")
    archive_path = work_dir / "downloads" / source["archive"]
    download_archive(source["url"], archive_path)
    raw_paths = extract_pair(archive_path, source["members"], work_dir / "raw" / source["name"])
    corpora = [{**source, "raw_paths": raw_paths}]

    tokenized_dir = work_dir / "tokenized"
    metadata["tokenization"]["eval"] = tokenize_split("eval", corpora, tokenized_dir, output_dir)
    codes_path = output_dir / metadata["tokenizer"]["bpe_codes"]
    bpe_dir = work_dir / "bpe"
    apply_joint_bpe(tokenized_dir, bpe_dir, codes_path, data_config["bpe_separator"], splits=("eval",))

    vocab_path = output_dir / metadata["tokenizer"]["vocab"]
    token_to_id = json.loads(vocab_path.read_text(encoding="utf-8"))["token_to_id"]
    metadata["splits"]["eval"] = save_split("eval", bpe_dir, output_dir, token_to_id, data_config)
    metadata["processing"]["eval_set"] = source["name"]
    metadata["processing"]["reference_files"]["eval"] = "eval/reference.de"
    metadata["data_config"] = data_config
    save_json(output_dir / "metadata.json", metadata)
    print(f"验证集准备完成：{output_dir / 'eval'}", flush=True)
    return metadata


"""
download_archive：下载原始语料，得到配置中指定的压缩包
"""
def download_archive(url: str, destination: Path):
    if destination.is_file():
        print(f"复用压缩包：{destination.name}", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    request = Request(url, headers={"User-Agent": "WMT14-Transformer-Preparation/1.0"})
    print(f"下载：{url}", flush=True)

    with urlopen(request, timeout=120) as response, temporary_path.open("wb") as file:
        expected_size = response.headers.get("Content-Length")
        shutil.copyfileobj(response, file, length=1024 * 1024)

    if expected_size is not None and temporary_path.stat().st_size != int(expected_size):
        raise ValueError(f"下载不完整：{destination.name}")
    temporary_path.replace(destination)


"""
extract_pair：只读取压缩包中指定的英德文件，不解压其他语言
输入：
    members：配置中指定的英德文件路径
返回：
    {语言: 原文文件路径}
"""
def extract_pair(archive_path: Path, members: dict, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    paths = {language: destination / Path(name).name for language, name in members.items()}
    if all(path.is_file() for path in paths.values()):
        return paths

    remaining = set(members.values())
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in remaining:
                continue

            # 使用配置中的文件名作为目标，不使用压缩包提供的目录进行解压
            destination_path = destination / Path(name).name
            temporary_path = destination_path.with_name(destination_path.name + ".part")
            with archive.extractfile(member) as source, temporary_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary_path.replace(destination_path)
            remaining.remove(name)
            if not remaining:
                break

    if remaining:
        raise FileNotFoundError(f"压缩包缺少配置中指定的文件：{sorted(remaining)}")
    return paths


"""
tokenize_split：标点规范化、Moses 分词，并保留大小写
输入：
    split：train、eval 或 test
    corpora：当前划分的原文文件与来源信息
    directory：分词结果目录
    output_dir：最终数据目录，eval、test 原文也保存到这里
返回：
    各个语料处理前后的句子对数量
"""
def tokenize_split(split: str, corpora: list, directory: Path, output_dir: Path) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    normalizers = {language: MosesPunctNormalizer(lang=language) for language in ("en", "de")}
    tokenizers = {language: MosesTokenizer(lang=language) for language in ("en", "de")}
    statistics = []

    with (
        (directory / f"{split}.en").open("w", encoding="utf-8") as source_file,
        (directory / f"{split}.de").open("w", encoding="utf-8") as target_file,
    ):
        for corpus in corpora:
            read_pairs = 0
            kept_pairs = 0
            raw_paths = corpus["raw_paths"]
            for source, target in read_parallel(raw_paths["en"], raw_paths["de"]):
                read_pairs += 1
                source_text = preprocess_sentence(source, normalizers["en"], tokenizers["en"])
                target_text = preprocess_sentence(target, normalizers["de"], tokenizers["de"])

                if not source_text.strip() or not target_text.strip():
                    if split == "train":
                        continue
                    raise ValueError(f"{split} 出现空句子，不能删除后改变评分样本：{corpus['name']}")

                source_file.write(source_text + "\n")
                target_file.write(target_text + "\n")
                kept_pairs += 1
                if read_pairs % 100000 == 0:
                    print(f"{corpus['name']}：已读取 {read_pairs:,} 对句子", flush=True)

            if "expected_pairs" in corpus and read_pairs != corpus["expected_pairs"]:
                raise ValueError(f"{split} 版本或句子数量与配置不一致：{corpus['name']}，实际 {read_pairs}")
            statistics.append({"name": corpus["name"], "read_pairs": read_pairs, "kept_pairs": kept_pairs})
            print(f"分词完成：{corpus['name']}，保留 {kept_pairs:,}/{read_pairs:,} 对", flush=True)

    if split != "train":
        save_originals(corpora, output_dir / split)
    return statistics


"""
read_parallel：同步读取一对英文和德文，防止长度不一致时静默丢失句子
返回：
    source 和 target 的生成器
"""
def read_parallel(source_path: Path, target_path: Path):
    for source, target in zip(read_sentences(source_path), read_sentences(target_path), strict=True):
        yield source, target


"""
read_sentences：读取逐行文本，或从 SGML 中取出句子
返回：
    每次生成一个句子，不把完整训练集读入列表
"""
def read_sentences(path: Path):
    if path.suffix == ".sgm":
        # SGML 用于较小的 eval、test；去掉标签，再还原 &amp; 等实体
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"<seg\b[^>]*>(.*?)</seg\s*>", text, re.IGNORECASE | re.DOTALL):
            sentence = re.sub(r"<[^>]+>", "", match.group(1))
            yield " ".join(html.unescape(sentence).split())
    else:
        # 只按 LF 分行，避免句子内部的回车字符改变英德行对应关系
        with path.open("r", encoding="utf-8", newline="\n") as file:
            for line in file:
                yield line.strip()


"""
preprocess_sentence：清理控制字符、规范标点，再调用 Moses 分词
输入：
    sentence：原始句子
    normalizer：Moses 标点规范化器
    tokenizer：Moses 分词器
返回：
    用空格分隔的 token 文本，保留大小写
"""
def preprocess_sentence(sentence: str, normalizer, tokenizer) -> str:
    sentence = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sentence)
    sentence = normalizer.normalize(sentence)
    return tokenizer.tokenize(sentence, aggressive_dash_splits=True, return_str=True, escape=True)


"""
save_originals：保存 eval、test 原文，供后续翻译和 BLEU 评分
"""
def save_originals(corpora: list, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (
        (directory / "source.en").open("w", encoding="utf-8") as source_file,
        (directory / "reference.de").open("w", encoding="utf-8") as target_file,
    ):
        for corpus in corpora:
            raw_paths = corpus["raw_paths"]
            for source, target in read_parallel(raw_paths["en"], raw_paths["de"]):
                source_file.write(source + "\n")
                target_file.write(target + "\n")


"""
learn_joint_bpe：合并英德训练文本，调用 subword-nmt 学习共同 BPE 规则
输入：
    directory：Moses 分词结果目录
    codes_path：BPE 规则输出路径
    data_config：BPE 合并次数、最低词频等配置
"""
def learn_joint_bpe(directory: Path, codes_path: Path, data_config: dict):
    # 在临时目录中依次写入两种语言的训练文本，不读入完整句子列表
    joint_path = directory / "train.en-de.txt"
    with joint_path.open("w", encoding="utf-8") as target:
        for language in ("en", "de"):
            with (directory / f"train.{language}").open("r", encoding="utf-8") as source:
                shutil.copyfileobj(source, target)

    print("学习英德联合 BPE，仅使用训练集", flush=True)
    # BPE 合并次数不等于最终词表大小，实际 V 在构建共同词表后确定
    with joint_path.open("r", encoding="utf-8") as source, codes_path.open("w", encoding="utf-8") as target:
        learn_bpe(
            source, target, num_symbols=data_config["bpe_merges"],
            min_frequency=data_config["bpe_min_frequency"],
        )


"""
apply_joint_bpe：各划分复用训练集学出的 BPE，不重新学习规则
输入：
    tokenized_dir：分词结果目录
    bpe_dir：子词结果目录
    codes_path：共同 BPE 规则
    separator：非结尾子词的标记，例如 @@
    splits：需要处理的划分，补充验证集时只传 ("eval",)
"""
def apply_joint_bpe(tokenized_dir: Path, bpe_dir: Path, codes_path: Path, separator: str, splits=("train", "eval", "test")):
    bpe_dir.mkdir(parents=True, exist_ok=True)
    with codes_path.open("r", encoding="utf-8") as codes:
        bpe = BPE(codes, separator=separator)  # 加载已经学好的规则
        for split in splits:
            for language in ("en", "de"):
                filename = f"{split}.{language}"
                print(f"应用 BPE：{filename}", flush=True)
                with (
                    (tokenized_dir / filename).open("r", encoding="utf-8") as source,
                    (bpe_dir / filename).open("w", encoding="utf-8") as target,
                ):
                    for line in source:
                        target.write(bpe.process_line(line).strip() + "\n")


"""
build_vocab：统计保留的训练句子，建立英德共同词表
输入：
    bpe_dir：子词结果目录
    data_config：特殊 token 与词表大小上限
    tokenizer_dir：共同词表输出目录
返回：
    token_to_id：{token: ID}，实际词表大小由统计结果决定
"""
def build_vocab(bpe_dir: Path, data_config: dict, tokenizer_dir: Path) -> dict:
    token_counts = Counter()
    for source_tokens, target_tokens in read_bpe_pairs(bpe_dir, "train", data_config):
        token_counts.update(source_tokens)
        token_counts.update(target_tokens)

    tokens = list(data_config["special_tokens"])
    for token in tokens:
        token_counts.pop(token, None)
    capacity = data_config["max_vocab_size"] - len(tokens)
    tokens.extend(token for token, _ in token_counts.most_common(capacity))

    token_to_id = {}
    for token in tokens:
        token_to_id[token] = len(token_to_id)
    save_json(tokenizer_dir / "vocab.json", {"tokens": tokens, "token_to_id": token_to_id})
    print(f"共同词表：{len(token_to_id):,} 个 token，包含特殊 token", flush=True)
    return token_to_id


"""
read_bpe_pairs：读取子词序列，仅对训练集应用长度与长度比过滤
输入：
    directory：子词结果目录
    split：当前数据划分
    data_config：训练集清理规则
返回：
    每次生成一对 token 列表，eval、test 不做长度裁剪或删句
"""
def read_bpe_pairs(directory: Path, split: str, data_config: dict):
    for source, target in read_parallel(directory / f"{split}.en", directory / f"{split}.de"):
        source_tokens = source.split()
        target_tokens = target.split()

        if split == "train":
            source_length = len(source_tokens)
            target_length = len(target_tokens)
            if max(source_length, target_length) > data_config["max_train_tokens"]:
                continue
            if max(source_length, target_length) / min(source_length, target_length) > data_config["max_length_ratio"]:
                continue
        yield source_tokens, target_tokens


"""
save_json：把字典写入 JSON 文件
"""
def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


"""
save_split：将子词转成 ID，达到分片大小后立即写入磁盘
输入：
    split：当前划分
    bpe_dir、output_dir：子词目录、最终数据目录
    token_to_id：共同词表
    data_config：清理规则与每个分片的句子对数量
返回：
    当前划分的样本数与分片清单
"""
def save_split(split: str, bpe_dir: Path, output_dir: Path, token_to_id: dict, data_config: dict) -> dict:
    source_sequences = []
    target_sequences = []
    shards = []
    num_pairs = 0
    unk_id = token_to_id["<unk>"]
    eos_id = token_to_id["<eos>"]

    for source_tokens, target_tokens in read_bpe_pairs(bpe_dir, split, data_config):
        source_ids = [token_to_id.get(token, unk_id) for token in source_tokens]
        target_ids = [token_to_id.get(token, unk_id) for token in target_tokens]
        source_sequences.append(source_ids + [eos_id])
        target_sequences.append(target_ids)
        num_pairs += 1

        if len(source_sequences) == data_config["shard_size"]:
            relative_path = Path(split) / f"shard-{len(shards):05d}"
            statistics = save_shard(output_dir / relative_path, source_sequences, target_sequences)
            shards.append({"path": relative_path.as_posix(), **statistics})
            source_sequences.clear()
            target_sequences.clear()
            print(f"保存 {split}：累计 {num_pairs:,} 对句子", flush=True)

    if source_sequences:
        relative_path = Path(split) / f"shard-{len(shards):05d}"
        statistics = save_shard(output_dir / relative_path, source_sequences, target_sequences)
        shards.append({"path": relative_path.as_posix(), **statistics})
    if num_pairs == 0:
        raise ValueError(f"处理后没有保留句子：{split}")
    return {"num_pairs": num_pairs, "shards": shards}


"""
save_shard：保存当前分片的 ID、句子边界和长度，供 data.py 按需读取
输入：
    shard_dir：分片目录
    source_sequences、target_sequences：当前分片的句子 ID 列表，尚未补齐
返回：
    当前分片的句子数与 token 数

N：当前分片的句子对数量
"""
def save_shard(shard_dir: Path, source_sequences: list, target_sequences: list) -> dict:
    shard_dir.mkdir(parents=True, exist_ok=True)
    statistics = {"num_pairs": len(source_sequences)}
    for side, sequences in (("source", source_sequences), ("target", target_sequences)):
        lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int32)  # [N]
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)  # [N + 1]
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)
        token_ids = np.fromiter(chain.from_iterable(sequences), dtype=np.int32, count=int(offsets[-1]))  # [该侧 token 总数]

        # 第 i 个句子的 ID 是 token_ids[offsets[i]:offsets[i + 1]]
        np.save(shard_dir / f"{side}_ids.npy", token_ids, allow_pickle=False)
        np.save(shard_dir / f"{side}_offsets.npy", offsets, allow_pickle=False)
        np.save(shard_dir / f"{side}_lengths.npy", lengths, allow_pickle=False)
        statistics[f"{side}_tokens"] = int(offsets[-1])
    return statistics


if __name__ == "__main__":
    main()
