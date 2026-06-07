import math
from typing import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 weight_decay: float = 0.01,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8):
        defaults = {
            'alpha': lr,
            'lamb': weight_decay,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group['alpha']
            lamb = group['lamb']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 1)
                grad = p.grad.data
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha * lamb * p.data
                m_prev = state.get('m', torch.zeros_like(p.data))
                v_prev = state.get('v', torch.zeros_like(p.data))
                state['m'] = beta1 * m_prev + (1 - beta1) * grad
                state['v'] = beta2 * v_prev + (1 - beta2) * torch.square(grad)
                p.data -= alpha_t * state['m'] / (torch.sqrt(state['v']) + eps)
                state['t'] = t + 1
        return loss
