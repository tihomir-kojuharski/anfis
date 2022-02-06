import torch


class SingletonMembershipFunction(torch.nn.Module):
    def __init__(self, value: float):
        super(SingletonMembershipFunction, self).__init__()

        self.__value = value

    def forward(self, x):
        return (x == self.__value).float()
