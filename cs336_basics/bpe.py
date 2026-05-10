import os

def train_bpe_tokenizer(input_path: str | os.PathLike,
                        vocab_size: int,
                        special_tokens: list) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    exists_token: set[bytes] = set(vocab.values())
    next_token_id = len(vocab)

    for special_token in special_tokens:
        if special_token not in exists_token:
            vocab[next_token_id] = special_token
            next_token_id += 1

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except FileNotFoundError:
        text = ''

    #todo 分割字符串
    raise NotImplementedError


if __name__ == '__main__':
    train_bpe_tokenizer("", 2, [])
