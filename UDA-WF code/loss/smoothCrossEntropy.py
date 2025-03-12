import torch
from torch import nn
from torch.nn import functional as F


class smoothCrossEntropy(nn.Module): #平滑交叉熵

    def __init__(self, *args, **kwargs) -> None:
        super(smoothCrossEntropy, self).__init__(*args, **kwargs)

    def forward(self, y_pred, y, ratio):
        b, K= y.shape
        y_exp = -F.log_softmax(y_pred, dim=-1) # softmax

        y_smooth = (1 - ratio) * y  + ratio/K

        loss = y_smooth * y_exp
        return loss.sum(axis=1).mean()
