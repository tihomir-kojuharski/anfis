from hamcrest import assert_that, equal_to, close_to

from anfis.membership_functions.gaussian_membership_function import GaussianMembershipFunction

eps = 1e-5


def test_case1():
    mf = GaussianMembershipFunction(5, 2)

    result = mf(7).item()
    assert_that(result, close_to(0.6065306663513184, eps))


def test_case2():
    mf = GaussianMembershipFunction(5, 2)

    result = mf(5).item()
    assert_that(result, equal_to(1))
