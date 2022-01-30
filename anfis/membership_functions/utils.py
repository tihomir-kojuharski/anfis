import torch

def create_parameter(value: float):
    return torch.nn.Parameter(torch.tensor(value, dtype=torch.float))