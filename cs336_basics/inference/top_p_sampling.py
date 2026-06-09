import torch
from torch import Tensor


def top_p_sampling(probs: Tensor, p: float = 0.9) -> Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # 计算累加
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # <=p的元素为True，其他为false
    mask = cumulative_probs <= p

    # 至少取一个token
    mask[..., 0] = True

    # 根据mask判断，为true用sorted_probs, false用torch.zeros_like(sorted_probs)
    filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))

    filtered_probs = filtered_probs/torch.sum(filtered_probs, dim=-1, keepdim=True)

    output_probs = torch.zeros_like(probs)
    output_probs.scatter_(-1, sorted_indices, filtered_probs)
    return output_probs