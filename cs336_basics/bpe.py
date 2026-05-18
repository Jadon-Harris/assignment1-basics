import os
import pickle
import time
from collections import defaultdict

import regex

from cs336_basics.log_utils import log_info


def get_word_one_by_one(file_path: str | os.PathLike, special_tokens: list[str], chunk_size: int = 64 * 1024 * 1024):
    # # 1. '(?:[sdmt]|ll|ve|re)    ：匹配英文缩写后缀，如 's、'd、'm、't、'll、've、're
    # # 2.  ?\p{L}+                ：匹配“可选前导空格 + 连续字母”，如 hello、 world、你好
    # # 3.  ?\p{N}+                ：匹配“可选前导空格 + 连续数字”，如 123、 2026
    # # 4.  ?[^\s\p{L}\p{N}]+      ：匹配“可选前导空格 + 连续符号/标点”，如 !、 !!!、<|endoftext|>
    # # 5. \s+(?!\S)               ：匹配一串空白，且后面不是非空白字符，通常用于匹配结尾空白
    # # 6. \s+                     ：匹配其他空白字符，如空格、换行、Tab
    pat = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    special_tokens_re = None
    if special_tokens:
        special_tokens_re = regex.compile('|'.join(map(regex.escape, special_tokens)))

    carry = ""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                if carry:
                    chunks = special_tokens_re.split(carry) if special_tokens_re else [carry]
                    for chunk in chunks:
                        for word in pat.findall(chunk):
                            yield word
                break
            text = carry + block
            cut = max(text.rfind(" "),
                      text.rfind("\r"),
                      text.rfind("\n"),
                      text.rfind("\t"))
            if cut == -1:
                cut = max(0, len(text) - 4096)

            process_text = text[:cut]
            carry = text[cut:]
            chunks = special_tokens_re.split(process_text) if special_tokens_re else [process_text]
            for chunk in chunks:
                for word in pat.findall(chunk):
                    yield word


def merge_token_sequence(token_seq: tuple, best_pair: tuple) -> tuple:
    new_seq = []
    combined_token = best_pair[0] + best_pair[1]
    i = 0
    while i < len(token_seq):
        # 检查当前位置是否是最佳对的开始
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == best_pair:
            new_seq.append(combined_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1
    return tuple(new_seq)


def train_bpe_tokenizer(input_path: str | os.PathLike,
                        vocab_size: int,
                        special_tokens: list) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    file_size = os.path.getsize(input_path) / 1024 / 1024 / 1024

    log_info(f"Start training BPE tokenizer")
    log_info(f"Input file: {input_path}")
    log_info(f"File size: {file_size:.2f} GB")
    log_info(f"Target vocab size: {vocab_size}")
    log_info(f"Special tokens: {special_tokens}")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    exists_token: set[bytes] = set(vocab.values())
    next_token_id = len(vocab)
    token_freq_table = defaultdict(int)

    # 将special token 加入vocab
    for special_token in special_tokens:
        if special_token not in exists_token:
            vocab[next_token_id] = special_token
            next_token_id += 1

    # ！！！ 内存不够，不能一次读取
    # # 读取文件
    # try:
    #     with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
    #         text = f.read()
    # except FileNotFoundError:
    #     text = ''
    #
    # # 分割字符串
    # # # 先按special token做分割，注意special token中存在一些特殊字符，需要进行转译
    # chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
    # # # 再按如下规则匹配切分:
    # # # 1. '(?:[sdmt]|ll|ve|re)    ：匹配英文缩写后缀，如 's、'd、'm、't、'll、've、're
    # # # 2.  ?\p{L}+                ：匹配“可选前导空格 + 连续字母”，如 hello、 world、你好
    # # # 3.  ?\p{N}+                ：匹配“可选前导空格 + 连续数字”，如 123、 2026
    # # # 4.  ?[^\s\p{L}\p{N}]+      ：匹配“可选前导空格 + 连续符号/标点”，如 !、 !!!、<|endoftext|>
    # # # 5. \s+(?!\S)               ：匹配一串空白，且后面不是非空白字符，通常用于匹配结尾空白
    # # # 6. \s+                     ：匹配其他空白字符，如空格、换行、Tab
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # for chunk in chunks:
    #     for word in regex.findall(PAT, chunk):
    #         word_bytes = word.encode('utf-8')
    #         bytes_list = [bytes([x]) for x in word_bytes]
    #         token_freq_table[tuple(bytes_list)] += 1

    for word in get_word_one_by_one(input_path, special_tokens, chunk_size=64 * 1024 * 1024):
        word_bytes = word.encode("utf-8")
        bytes_list = [bytes([x]) for x in word_bytes]
        token_freq_table[tuple(bytes_list)] += 1

    # 统计pair出现次数
    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    for token in token_freq_table.keys():
        for i in range(len(token) - 1):
            pair_counts[token[i], token[i + 1]] += token_freq_table[token]

    merges: list[tuple[bytes, bytes]] = []

    merge_start_time = time.time()
    merge_step = 0
    log_every_merges = 100

    log_info("Start BPE merge loop")
    while len(vocab) < vocab_size:
        if not pair_counts:
            log_info("pair_counts is empty, stop training")
            break

        merge_step += 1

        # 找到频率最高的，可能有多个最多的次数相同
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        # 选一个字节序最大的，不是字节序最大效果最好，而是为了统一标准，保证每次训练的结果一致
        best_pair = max(candidates)
        # 将这个合并规则添加到merges
        merges.append(best_pair)

        # 合并为新token，添加到vocab
        combined_token = best_pair[0] + best_pair[1]
        vocab[next_token_id] = combined_token
        next_token_id += 1

        # 查看token_freq_table有哪些token中包含best_pair
        affected_tokens = []
        for token, freq in token_freq_table.items():
            if any(token[i:i + 2] == best_pair for i in range(len(token) - 1)):
                affected_tokens.append((token, freq))

        for token, freq in affected_tokens:
            # 将token中各个pair贡献的值从pair_counts中减掉
            for i in range(len(token) - 1):
                pair_counts[token[i], token[i + 1]] -= freq
                if pair_counts[token[i], token[i + 1]] <= 0:
                    pair_counts.pop((token[i], token[i + 1]), None)

            # 合并best_pair，并生成一个新的token
            new_token_bytes = merge_token_sequence(token, best_pair)
            # 更新pairs
            for i in range(len(new_token_bytes) - 1):
                pair = (new_token_bytes[i], new_token_bytes[i + 1])
                pair_counts[pair] += freq

            token_freq_table.pop(token, None)
            token_freq_table[new_token_bytes] += freq
        if merge_step % log_every_merges == 0 or len(vocab) == vocab_size:
            elapsed = time.time() - merge_start_time
            avg_time = elapsed / merge_step if merge_step > 0 else 0

            log_info(
                f"merge step: {merge_step:,}, "
                f"vocab size: {len(vocab):,}/{vocab_size:,}, "
                f"best_pair: {best_pair}, "
                f"count: {max_count:,}, "
                f"affected token seqs: {len(affected_tokens):,}, "
                f"token_freq_table size: {len(token_freq_table):,}, "
                f"pair_counts size: {len(pair_counts):,}, "
                f"avg merge time: {avg_time:.4f}s"
            )
    # 保存词汇表到文件，使用二进制写入模式
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # 保存 BPE 合并规则到文件，使用二进制写入模式
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges


if __name__ == '__main__':
    special_tokens = ["<|endoftext|>"]
    # vocab, merges = train_bpe_tokenizer("../data/TinyStoriesV2-GPT4-train.txt", 5000, special_tokens)
    vocab, merges = train_bpe_tokenizer("../data/test.txt", 5000, special_tokens)
    log_info(f"vocab size: {len(vocab):,}")
    log_info(f"merges size: {len(merges):,}")