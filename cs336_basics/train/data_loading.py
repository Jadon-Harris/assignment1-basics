import numpy.typing as npt
import torch


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
              ) -> tuple[torch.Tensor, torch.Tensor]:
    start_indexes = torch.randint(low=0, high=dataset.size - context_length, size=(batch_size,))
    x = [dataset[i: i + context_length] for i in start_indexes]
    y = [dataset[i + 1: i + context_length + 1] for i in start_indexes]
    return torch.LongTensor(x).to(device=device), torch.LongTensor(y).to(device=device)
