import torch
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import nn, Tensor

from cs336_basics.module.softmax import softmax


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                q: Float[Tensor, " ... queries d_k"],
                k: Float[Tensor, " ... keys d_k"],
                v: Float[Tensor, " ... keys d_v"],
                mask: Bool[Tensor, " ... queries keys"] | None = None,
                ) -> Float[Tensor, " ... queries d_v"]:
        q_k_score = einsum(q, k, '... queries d_k, ... keys d_k -> ... queries keys') * q.size(-1) ** (-0.5)
        if mask is not None:
            q_k_score = q_k_score.masked_fill(mask == False, float('-inf'))
        q_k_attention = softmax(q_k_score, dim=-1)
        return einsum(q_k_attention, v, '... queries keys, ... keys d_v -> ... queries d_v')