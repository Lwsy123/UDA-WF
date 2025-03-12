import torch 
from torch import nn 
from torch.nn import functional as F

class WeightedCrossEntropy(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropy,self).__init__()

    def forward(self, y_pre, y_label, weighted = None):
        _, K = y_label.shape
        y_exp = F.softmax(y_pre, dim=-1)

        CE = -y_label*torch.log(y_exp)
        if weighted is not None:
            CE = (CE.sum(axis =-1) )*weighted

        return CE.mean()
        