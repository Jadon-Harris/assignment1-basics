import regex
from cs336_basics.bpe_trainer import train_bpe_tokenizer


class BpeTokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        if special_tokens:
            special_tokens_pat = '|'.join(
                regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            self.special_tokens_reg = regex.compile(f'({special_tokens_pat})')
        else:
            self.special_tokens_reg = None

    def encode(self, text: str) -> list[int]:
        token_ids = []

        if self.special_tokens_reg is None:
            token_ids.extend(self._encode_ordinary(text))

        chunks = self.special_tokens_reg.split(text)
        for chunk in chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                token_ids.extend(self._encode_ordinary(chunk))
        return token_ids

    def _encode_ordinary(self, text: str) -> list[int]:
        token_ids = []
        pat = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        pre_tokens = pat.findall(text)
        for pre_token in pre_tokens:
            byte_tokens = [bytes([x]) for x in pre_token.encode("utf-8")]
            bpe_tokens = self.merge_sub_tokens(byte_tokens)

            for token in bpe_tokens:
                token_ids.append(self.token_to_id[token])

        return token_ids

    def merge_sub_tokens(self, tokens: list[bytes]) -> list[bytes]:
        while len(tokens) >= 2:
            best_pair = None
            best_rank = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)

                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        token_bytes: bytes = b''.join(self.vocab[token_id] for token_id in token_ids)
        return token_bytes.decode("utf-8", errors='replace')


if __name__ == '__main__':
    text = "hello world <|endoftext|> hahahah"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe_tokenizer("../data/test.txt", 300, special_tokens)
    tokenizer = BpeTokenizer(vocab, merges, special_tokens)
    res = tokenizer.encode(text)
    assert text == tokenizer.decode(res)
