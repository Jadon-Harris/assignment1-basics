import argparse

import torch
from torch import nn

from cs336_basics.bpe.bpe_tokenizer import BpeTokenizer
from cs336_basics.config.config import MERGES_PATH, CHECKPOINT_PATH, VOCAB_PATH
from cs336_basics.inference.softmax_with_temperature import softmax_with_temperature
from cs336_basics.inference.top_p_sampling import top_p_sampling
from cs336_basics.transformer.transformer_lm import TransformerLM


def get_args():
    parser = argparse.ArgumentParser()
    # model and tokenizer
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='path to model checkpoint')
    parser.add_argument('--vocab', type=str, default=VOCAB_PATH, help='path to vocabulary file')
    parser.add_argument('--merges', type=str, default=MERGES_PATH, help='path to merges file')

    # generate
    parser.add_argument('--prompt', type=str, default='Once upon a time', help='input prompt')
    parser.add_argument('--max_tokens', type=int, default=256, help='max number of tokens')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for sampling (lower = more deterministic)')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p for nucleus sampling')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')

    # device
    parser.add_argument('--device', type=str, default='auto', help='device to use')

    # end of sequence token
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>', help='eos token')

    return parser.parse_args()


def get_generate_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def load_model_and_tokenizer(checkpoint_path: str, vocab_path: str, merges_path: str, device: str = 'cpu'):
    tokenizer = BpeTokenizer.from_files(vocab_path, merges_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg = {
        'vocab_size': 10000,
        'context_length': 256,
        'num_layers': 4,
        'num_heads': 16,
        'd_model': 512,
        'd_ff': 1344,
        'rope_theta': 10000.0
    } if len(checkpoint['model_config']) == 0 else checkpoint['model_config']

    model = TransformerLM(
        vocab_size=int(cfg.get('vocab_size', 10000)),
        context_length=int(cfg.get('context_length', 256)),
        num_layers=int(cfg.get('num_layers', 4)),
        d_model=int(cfg.get('d_model', 512)),
        num_heads=int(cfg.get('num_heads', 16)),
        rope_theta=float(cfg.get('rope_theta', 10000.0)),
        d_ff=int(cfg.get('d_ff', 1344))).to(device)

    model.load_state_dict(checkpoint["model"])

    return model, tokenizer


def generate_text(model: nn.Module,
                  tokenizer,
                  prompt: str,
                  max_tokens: int = 256,
                  temperature: float = 1.0,
                  top_p: float = 0.9,
                  device: str = 'auto',
                  eos_token: str = '<|endoftext|>') -> str:
    model.eval()

    # prompt_tokens (sequence_size,)
    prompt_tokens = tokenizer.encode(prompt)

    # input_ids: (batch, sequence_size)
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens = prompt_tokens.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)

            next_token_logits = logits[0, -1, :]

            probs = softmax_with_temperature(next_token_logits, temperature)

            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)

            # sample next token, return the token id
            next_token = torch.multinomial(probs, 1).item()

            generated_tokens.append(next_token)

            if eos_token is not None:
                decoded_token = tokenizer.decode([next_token])
                if decoded_token.strip() == eos_token.strip():
                    break

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

            if input_ids.size(1) > model.context_length:
                input_ids = input_ids[:, -model.context_length:]
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def main():
    args = get_args()
    device = get_generate_device(args.device)
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.vocab, args.merges, device)

    print(f"model has {sum(p.numel() for p in model.parameters())} parameters")

    print(f"\nGenerating {args.num_samples} samples with prompt: '{args.prompt}'")
    print(f"max tokens: {args.max_tokens}, temperature: {args.temperature}")
    print("-" * 80)
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n Sample {i + 1}:")
            print('-' * 40)
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                eos_token=args.eos_token
            )

            print(generated_text)
            print('=' * 80)
        except Exception as e:
            print(f'Error: {e}')
            break


if __name__ == '__main__':
    main()
