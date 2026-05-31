import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_type = x.dtype
        x = x.to(dtype=torch.float32)
        r_rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        rms_x = x * r_rms
        return einsum(rms_x, self.weight, "... d_model, d_model -> ... d_model").to(dtype=input_type)
