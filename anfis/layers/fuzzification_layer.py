from typing import List, Dict

import torch

from anfis.fuzzy_variable import FuzzyVariable


class FuzzificationLayer(torch.nn.Module):
    def __init__(self, variables: List[FuzzyVariable]):
        super(FuzzificationLayer, self).__init__()

        self.variables = torch.nn.ModuleList(variables)

        max_mfs_count = max([variable.mfs_count for variable in variables])
        for variable in self.variables:
            variable.pad_to(max_mfs_count)

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == len(self.variables)

        return torch.stack([self.variables[i](x[:, i:i + 1]) for i in range(len(self.variables))], dim=1)
