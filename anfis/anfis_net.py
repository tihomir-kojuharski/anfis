from collections import OrderedDict
from typing import List

import torch.nn

from anfis.fuzzy_variable import FuzzyVariable
from anfis.layers.consequent_layer import ConsequentLayer
from anfis.layers.fuzzification_layer import FuzzificationLayer
from anfis.layers.normalization_layer import NormalizationLayer
from anfis.layers.premise_layer import PremiseLayer
from anfis.layers.summing_layer import SummingLayer


class AnfisNet(torch.nn.Module):
    def __init__(self, variables: List[FuzzyVariable], out_variables: int, head_activation=None):
        super(AnfisNet, self).__init__()

        premise_layer = PremiseLayer(variables)

        self.layers = torch.nn.ModuleDict(OrderedDict([
            ('fuzzification', FuzzificationLayer(variables)),
            ('premise', premise_layer),
            ('normalization', NormalizationLayer()),
            ('consequent', ConsequentLayer(len(variables), premise_layer.rules_count, out_variables)),
            ('summing', SummingLayer(head_activation))]))

    def forward(self, x):
        x_orig = x

        x = self.layers['fuzzification'](x)
        x = self.layers['premise'](x)

        x_rules_strength = self.layers['normalization'](x)
        x_rules_output = self.layers['consequent'](x_orig)

        output = self.layers['summing'](x_rules_strength, x_rules_output)

        return output

    def extra_repr(self):
        rstr = []
        variables = self.layers['fuzzification'].variables
        rule_ants = self.layers['premise'].extra_repr(variables).split('\n')
        for i, crow in enumerate(self.layers['consequent'].coeffs):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' '*9+'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)
