import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"],
                  targets: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    get_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    logexpsum = torch.logsumexp(inputs, dim=-1, keepdim=True)
    loss = -get_logits + logexpsum
    return loss.mean()