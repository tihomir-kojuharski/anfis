import math
from typing import Iterable
import numpy as np
import torch

from anfis.membership_functions.utils import create_parameter


class BooleanMembershipFunction(torch.nn.Module):
    def __init__(self, value: bool):
        super(BooleanMembershipFunction, self).__init__()

        self.__value = value

    def forward(self, x):
        return (x == self.__value).float()

    @staticmethod
    def get_all():
        return [BooleanMembershipFunction(True), BooleanMembershipFunction(False)]
