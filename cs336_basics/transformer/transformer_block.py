import torch
from torch import nn

from cs336_basics.module.multi_head_self_attention import MultiHeadSelfAttention
from cs336_basics.module.rms_norm import RMSNorm
from cs336_basics.module.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 device=None,
                 dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)

        if hasattr(self.attention, 'pe') and hasattr(self.attention.pe, 'inv_freq'):
            buf = self.attention.pe.inv_freq
            try:
                del self.attention.pe._buffers['inv_freq']
                self.attention.pe.register_buffer('inv_freq', buf, persistent=False)
            except Exception:
                pass

        self.ffn = SwiGLU(d_model, d_ff)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        # rm rms
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x
