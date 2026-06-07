import os
import pathlib
import pickle

from cs336_basics.bpe.bpe_tokenizer import BpeTokenizer
from cs336_basics.bpe.bpe_trainer import train_bpe_tokenizer, encode_txt_as_np_array
from cs336_basics.log.log_utils import log_info

TRAIN_TXT_DATA_PATH = '../../data/TinyStoriesV2-GPT4-train.txt'
VALID_TXT_DATA_PATH = '../../data/TinyStoriesV2-GPT4-valid.txt'
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

VOCAB_PATH = '../../tokenizer/tinystories_bep_vocab.pkl'
MERGES_PATH = '../../tokenizer/tinystories_bep_merges.pkl'

TRAIN_DATA_PATH = '../../data/train.dat'
VALID_DATA_PATH = '../../data/valid.dat'


if __name__ == '__main__':
    # train
    vocab, merges = train_bpe_tokenizer(TRAIN_TXT_DATA_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
    log_info(f"vocab size: {len(vocab):,}")
    log_info(f"merges size: {len(merges):,}")
    # 保存词汇表到文件，使用二进制写入模式
    os.makedirs(pathlib.Path(VOCAB_PATH).parent, exist_ok=True)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    # 保存 BPE 合并规则到文件，使用二进制写入模式
    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)

    # txt to numpy array
    tokenizer = BpeTokenizer(vocab, merges, SPECIAL_TOKENS)
    encode_txt_as_np_array(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
    encode_txt_as_np_array(tokenizer, VALID_TXT_DATA_PATH, VALID_DATA_PATH)