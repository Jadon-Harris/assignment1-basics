import math

import torch
import torch.nn as nn
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        sigma = math.sqrt(2 / (d_model + d_ff))
        nn.init.trunc_normal_(self.w1_weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        nn.init.trunc_normal_(self.w2_weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        nn.init.trunc_normal_(self.w3_weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3_x = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1_x * torch.sigmoid(w1_x)
        silu_w3_x = silu * w3_x
        return einsum(self.w2_weight, silu_w3_x, "d_model d_ff, ... d_ff -> ... d_model")
