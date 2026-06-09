import torch
from torch import Tensor


def softmax_with_temperature(logits: Tensor, temperature: float = 1.0) -> Tensor:
    scaled_logits = logits / temperature
    max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(scaled_logits - max_logits)
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

    return probs
