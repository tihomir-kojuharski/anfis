import torch
import torch.nn.functional


class NormalizationLayer(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=1, dim=1)
