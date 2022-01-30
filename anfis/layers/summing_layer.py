import torch


class SummingLayer(torch.nn.Module):
    def __init__(self, activation=None):
        super(SummingLayer, self).__init__()

        self.__activation = activation

    def forward(self, x_rules_strength, x_rules_output):
        output = x_rules_strength.reshape(x_rules_output.shape).mul(x_rules_output).sum(dim=2).squeeze(dim=1)
        if self.__activation:
            output = self.__activation(output)

        return output
