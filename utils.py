from enum import Enum

import torch

from anfis.anfis_net import AnfisNet
from anfis.fuzzy_variable import FuzzyVariable
from anfis.membership_functions.singleton_membership_function import SingletonMembershipFunction
from anfis.membership_functions.trapezoidal_membership_function import TrapezoidalMembershipFunction


class DS(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def get_weather_anfis_model(pretrained_weights_filename=None):
    variables = [
        FuzzyVariable("MinTemp", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("MaxTemp", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("Rainfall", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("WindGustSpeed", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("Humidity3pm", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("Pressure3pm", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("RainToday", [SingletonMembershipFunction(0), SingletonMembershipFunction(1)]),
    ]

    model = AnfisNet(variables, 1, head_activation=torch.nn.Sigmoid())

    if pretrained_weights_filename:
        model.load_state_dict(torch.load(pretrained_weights_filename))

    return model
