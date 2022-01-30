import math
from typing import Iterable

import torch

from anfis.membership_functions.utils import create_parameter


class GaussianMembershipFunction(torch.nn.Module):
    def __init__(self, mu: float, std: float):
        super(GaussianMembershipFunction, self).__init__()

        self.register_parameter('mu', create_parameter(torch.tensor(mu)))
        self.register_parameter('std', create_parameter(torch.tensor(std)))

    def forward(self, x):
        return torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.std ** 2))

    @staticmethod
    def get_functions(std: float, mus: Iterable[float]):
        return [GaussianMembershipFunction(mu, std) for mu in mus]
