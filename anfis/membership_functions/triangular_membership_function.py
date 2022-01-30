import torch

from anfis.membership_functions.utils import create_parameter


class TriangularMembershipFunction(torch.nn.Module):
    def __init__(self, start: float, peak: float, end: float):
        super(TriangularMembershipFunction, self).__init__()

        self.register_parameter('start', create_parameter(start))
        self.register_parameter('peak', create_parameter(peak))
        self.register_parameter('end', create_parameter(end))

    def forward(self, x):
        return torch.max(torch.min((x - self.start) / (self.peak - self.start),
                                   (self.end - x) / (self.end - self.peak)),
                         torch.zeros_like(x))
