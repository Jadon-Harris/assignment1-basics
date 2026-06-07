from typing import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    l2_norms = torch.zeros((), device=grads[0].device)
    for g in grads:
        l2_norms += torch.sum(g.data ** 2)
    l2_norms = torch.sqrt(l2_norms)
    if l2_norms > max_l2_norm:
        for g in grads:
            g.data *= max_l2_norm / (l2_norms+eps)
