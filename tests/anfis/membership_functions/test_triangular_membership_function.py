import torch
from hamcrest import assert_that, equal_to

from anfis.membership_functions.triangular_membership_function import TriangularMembershipFunction


def test_peak():
    mf = TriangularMembershipFunction(0, 5, 10)

    result = mf(torch.tensor([5, 0, 10, 7])).tolist()

    assert_that(result, equal_to([1, 0, 0, 0.6000000238418579]))