import math

import torch
from hamcrest import assert_that, equal_to

from anfis.fuzzy_variable import FuzzyVariable
from anfis.layers.premise_layer import PremiseLayer
from anfis.membership_functions.triangular_membership_function import TriangularMembershipFunction


def test_shape():
    samples_count = 10
    variables_count = 6
    mfs_count = 5

    variable = FuzzyVariable('x1', [TriangularMembershipFunction(0, 5, 10)] * mfs_count)

    layer = PremiseLayer([variable] * variables_count)

    x = torch.randn(samples_count, variables_count, mfs_count)

    result = layer(x)

    assert_that(result.shape, equal_to((samples_count, math.pow(mfs_count, variables_count))))
