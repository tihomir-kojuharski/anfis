from collections import OrderedDict

import torch
from hamcrest import assert_that, equal_to

from anfis.fuzzy_variable import FuzzyVariable
from anfis.membership_functions.triangular_membership_function import TriangularMembershipFunction


def test_shape():
    samples_count = 10
    mfs_count = 5

    variable = FuzzyVariable('x1', [TriangularMembershipFunction(0, 5, 10)] * mfs_count)

    x = torch.randn(samples_count, 1)
    result = variable(x)

    assert_that(result.shape, equal_to((samples_count, mfs_count)))


def test_list_of_mfs():
    variable = FuzzyVariable('x1', [TriangularMembershipFunction(0, 5, 10)])
    result = variable(torch.tensor([[5], [0], [10], [7]])).tolist()

    assert_that(result, equal_to([[1], [0], [0], [0.6000000238418579]]))


def test_mfs_count():
    variable1 = FuzzyVariable('x1', [TriangularMembershipFunction(0, 5, 10)])
    variable2 = FuzzyVariable('x2', [TriangularMembershipFunction(0, 5, 10), TriangularMembershipFunction(0, 4, 10)])
    variable3 = FuzzyVariable('x3', OrderedDict([
        ('small', TriangularMembershipFunction(0, 5, 10)),
        ('large', TriangularMembershipFunction(5, 10, 20)),
    ]))

    assert_that(variable1.mfs_count, equal_to(1))
    assert_that(variable2.mfs_count, equal_to(2))
    assert_that(variable3.mfs_count, equal_to(2))
