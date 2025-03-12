import torch
from torch import nn
from torch.nn import functional as f
from d2l import torch as d2l

class IdentiLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(IdentiLoss, self).__init__(*args, **kwargs)
    
    def forward(self, y_true):
        return y_true.mean()
        