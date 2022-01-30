import torch

from anfis.membership_functions.utils import create_parameter


class TrapezoidalMembershipFunction(torch.nn.Module):
    __pseudo_infinity = 10000000000

    def __init__(self, a: float, b: float, c: float, d: float, freeze_left=False, freeze_right=False):
        super(TrapezoidalMembershipFunction, self).__init__()

        if freeze_left:
            self.a = a
            self.b = b
        else:
            self.register_parameter('a', create_parameter(a))
            self.register_parameter('b', create_parameter(b))

        if freeze_right:
            self.c = c
            self.d = d
        else:
            self.register_parameter('c', create_parameter(c))
            self.register_parameter('d', create_parameter(d))

    def forward(self, x):
        return torch.max(
            torch.min(
                torch.min((x - self.a) / (self.b - self.a),
                          torch.ones_like(x)),
                (self.d - x) / (self.d - self.c)),
            torch.zeros_like(x))

    @staticmethod
    def get_range_functions():
        return [TrapezoidalMembershipFunction.get_left_open(-0.5, 0),
                TrapezoidalMembershipFunction(-1, -0.5, 0.5, 1),
                TrapezoidalMembershipFunction.get_right_open(0, 0.5)]

    @staticmethod
    def get_left_open(c: float, d: float):
        return TrapezoidalMembershipFunction(-(TrapezoidalMembershipFunction.__pseudo_infinity + 1),
                                             -TrapezoidalMembershipFunction.__pseudo_infinity,
                                             c, d, freeze_left=True)

    @staticmethod
    def get_right_open(a: float, b: float):
        return TrapezoidalMembershipFunction(a, b,
                                             TrapezoidalMembershipFunction.__pseudo_infinity,
                                             TrapezoidalMembershipFunction.__pseudo_infinity + 1,
                                             freeze_right=True)
