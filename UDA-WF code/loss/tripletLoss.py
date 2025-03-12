import torch
from torch import nn
from torch.nn import functional as f
from d2l import torch as d2l
import math

class TriLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(TriLoss, self).__init__(*args, **kwargs)
    
    def forward(self, X, alpha_value):
        _alpha = alpha_value
        positive_sim, negative_sim = X
        value = negative_sim - positive_sim + _alpha

        loss = torch.where(value > 0.0, value, 0.0)
        return loss.mean()


