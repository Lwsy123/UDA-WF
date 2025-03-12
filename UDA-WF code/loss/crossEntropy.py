import torch 
from torch import nn 
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy,self).__init__()

    def forward(self, y_pre, y_label):
        _, K = y_label.shape
        y_exp = F.softmax(y_pre, dim=-1)

        CE = -y_label*torch.log(y_exp)
        CE = CE.sum(axis =-1) 

        return CE.mean()