import itertools
from typing import List

import torch.nn

from anfis.fuzzy_variable import FuzzyVariable


class PremiseLayer(torch.nn.Module):
    def __init__(self, variables: List[FuzzyVariable]):
        super(PremiseLayer, self).__init__()

        mf_indices = itertools.product(*[range(variable.mfs_count) for variable in variables])
        self.__mf_indices = torch.tensor(list(mf_indices))

    @property
    def rules_count(self):
        return self.__mf_indices.shape[0]

    def forward(self, x):
        batch_indices = self.__mf_indices.expand((x.shape[0], -1, -1))
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        rules = torch.prod(ants, dim=2)

        return rules

    def extra_repr(self, variables=None):
        if not variables:
            return f"Rules count: {len(self.__mf_indices)}"
        row_ants = []
        mf_count = [var.mfs_count for var in variables]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for variable, i in zip(variables, rule_idx):
                thisrule.append('{} is {}'
                                .format(variable.name, list(variable.membership_functions.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)
