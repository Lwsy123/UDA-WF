import torch
from torch import nn
from torch.nn import functional as F

def accurate(y_pred, y):

    indexes = torch.argmax(y_pred, dim=-1)
    # print(indexes.shape)

    res = 0
    for i, index in enumerate(indexes):
        res += y[i, index]
    
    return res

