import torch
from hamcrest import assert_that, equal_to

from anfis.fuzzy_variable import FuzzyVariable
from anfis.layers.fuzzification_layer import FuzzificationLayer
from anfis.membership_functions.gaussian_membership_function import GaussianMembershipFunction
from anfis.membership_functions.triangular_membership_function import TriangularMembershipFunction


def test_shape_simple():
    samples_count = 10
    mfs_count = 5
    variables_count = 6

    variable = FuzzyVariable('x1', [TriangularMembershipFunction(0, 5, 10)] * mfs_count)

    layer = FuzzificationLayer([variable] * variables_count)

    x = torch.randn(samples_count, variables_count)
    result = layer(x)

    assert_that(result.shape, equal_to((samples_count, variables_count, mfs_count)))


def test_shape_complex():
    samples_count = 10

    variable1 = FuzzyVariable('x1', [TriangularMembershipFunction(0, 5, 10), TriangularMembershipFunction(5, 10, 20)])
    variable2 = FuzzyVariable('x2', GaussianMembershipFunction.get_functions(3, [0, 10, 20, 30, 40]))

    layer = FuzzificationLayer([variable1, variable2])

    x = torch.randn(samples_count, 2)
    result = layer(x)

    assert_that(result.shape, equal_to((samples_count, 2, 5)))