import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.module.linear import Linear
from cs336_basics.module.rope import RoPE
from cs336_basics.module.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int,
                 num_heads: int,
                 theta: float = None,
                 max_seq_len: int = None,
                 token_positions: Int[Tensor, " ... sequence_length"] | None = None,
                 pe: nn.Module = RoPE,
                 use_causal_mask: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_w = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_w = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_w = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_w = Linear(d_model, d_model, device=device, dtype=dtype)
        self.pe = None
        if pe is not None and theta is not None and max_seq_len is not None:
            self.pe = pe(theta, self.d_k, max_seq_len, device=device, dtype=dtype)
        self.token_positions = token_positions
        self.use_causal_mask = use_causal_mask

    def causal_mask(self, seq_len: int) -> Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        # 返回扩充两个维度，最终返回 (1, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[
        Tensor, " ... sequence_length d_model"]:
        # 计算Q, K, V
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.v_w(x)
        # 分头
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        # 旋转位置编码
        if self.pe is not None:
            q = self.pe(q, self.token_positions)
            k = self.pe(k, self.token_positions)
        # mask
        mask = None
        if self.use_causal_mask:
            mask = self.causal_mask(q.size(-2))
            mask = mask.to(device=q.device)
        # attention
        attention = ScaledDotProductAttention().forward(q, k, v, mask=mask)
        attention = rearrange(attention, "b h s d -> b s (h d)", h=self.num_heads)
        out = self.o_w(attention)
        return out
