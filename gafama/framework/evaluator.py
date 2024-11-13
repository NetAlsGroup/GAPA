import torch
from torch import nn


class BasicEvaluator(nn.Module):
    def __init__(self, pop_size, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pop_size = pop_size
        self.device = device

    def forward(self, **kwargs) -> torch.Tensor:
        pass

    def _to_device(self, device):
        pass
