import itertools

import torch.nn


class ConsequentLayer(torch.nn.Module):
    def __init__(self, variables_count, rules_count, output_vars_count):
        super(ConsequentLayer, self).__init__()
        shape = torch.Size([rules_count, output_vars_count, variables_count + 1])

        coeffs = torch.zeros(shape, dtype=torch.float, requires_grad=True)
        torch.nn.init.xavier_uniform_(coeffs)
        self.register_parameter('coeffs', torch.nn.Parameter(coeffs))

    def forward(self, x):
        x_with_ones = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        return torch.matmul(self.coeffs, x_with_ones.t()).transpose(0, 2)
