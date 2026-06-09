import os
from typing import BinaryIO, IO

import torch

from cs336_basics.transformer.transformer_lm import TransformerLM


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    if isinstance(model, TransformerLM):
        model_config = {
            'vocab_size': model.vocab_size,
            'context_length': model.context_length,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'num_layers': model.num_layers,
            'd_ff': model.d_ff,
            'rope_theta': model.rope_theta
        }
    else:
        model_config = {}

    state_dict = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "model_config": model_config,
                  "iteration": iteration}
    torch.save(state_dict, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]
