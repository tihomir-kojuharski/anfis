import torch
from hamcrest import assert_that, equal_to

from anfis.membership_functions.singleton_membership_function import SingletonMembershipFunction
from anfis.membership_functions.triangular_membership_function import TriangularMembershipFunction


def test_true():
    mf = SingletonMembershipFunction(1)
    result_true = mf(torch.tensor([1])).tolist()
    result_false = mf(torch.tensor([0])).tolist()

    assert_that(result_true, equal_to([1]))
    assert_that(result_false, equal_to([0]))