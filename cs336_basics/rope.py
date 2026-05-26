import torch
import torch.nn as nn
from einops import einsum, rearrange


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_buffer(name='inv_freq', tensor = 1 / (theta ** (torch.arange(0, d_k, 2).to(dtype=torch.float)/d_k)))  # (d_k/2, )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        elif token_positions.dim() == 2:
            token_positions = token_positions[0]

        theta = einsum(token_positions, self.inv_freq, 'n, d -> n d')  # (seq_len, d_k/2)

        cos = theta.cos().repeat_interleave(2, dim=-1).unsqueeze(0)  # (seq_len, d_k)
        sin = theta.sin().repeat_interleave(2, dim=-1).unsqueeze(0)  # (seq_len, d_k)

        rotated_x = self.rotate(x)

        return x * cos + rotated_x * sin

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, '... (s r) -> ... s r', r = 2)
        x_odd, x_even = x.unbind(dim=-1)
        x = torch.stack((-x_even, x_odd), dim=-1)
        return rearrange(x, '... s r -> ... (s r)')
