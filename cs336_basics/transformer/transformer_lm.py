import torch
from torch import nn

from cs336_basics.module import embedding
from cs336_basics.module.linear import Linear
from cs336_basics.module.rms_norm import RMSNorm
from cs336_basics.transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 d_ff: int,
                 rope_theta: float
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.embedding = embedding.Embedding(vocab_size, d_model)
        self.tf_layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.tf_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.linear(x)
        return x
