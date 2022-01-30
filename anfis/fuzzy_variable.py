from collections import OrderedDict
from typing import List, Union, Dict

import torch


class FuzzyVariable(torch.nn.Module):
    def __init__(self, name: str, mfs: Union[List[torch.nn.Module], Dict[str, torch.nn.Module]]):
        super(FuzzyVariable, self).__init__()

        self.__name = name

        mfs_dict = None
        if isinstance(mfs, list):
            names = [f'mf{i}' for i in range(len(mfs))]
            mfs_dict = OrderedDict(zip(names, mfs))
        else:
            mfs_dict = mfs

        self.__mfs = torch.nn.ModuleDict(mfs_dict)

        self.__padding = 0

    def pad_to(self, size: int):
        self.__padding = size - len(self.__mfs)

    @property
    def mfs_count(self):
        return len(self.__mfs)

    @property
    def membership_functions(self):
        return self.__mfs

    @property
    def name(self):
        return self.__name

    def forward(self, x):
        result = torch.cat([mf(x) for mf in self.__mfs.values()], dim=1)
        if self.__padding > 0:
            result = torch.cat([result, torch.zeros(x.shape[0], self.__padding)], dim=1)

        return result

    def members(self):
        return self.__mfs.items()
