import torch
from torch import nn
from torch.nn import functional as F


def transpose_qkv(data, head): #多头转换

    b,c, h = data.shape

    data = data.reshape(b,c, head, h//head)

    data = data.transpose(1,2)

    return data.reshape(-1, c, data.shape[-1])

def transpose_output(data, head):

    b,c, h = data.shape 
    data = data.reshape(b // head, head, c, -1)

    data = data.transpose(1,2)
    return data.reshape(b // head, c, -1)


def accurate(y_pred, y):

    indexes = torch.argmax(y_pred, dim=-1)
    # print(indexes.shape)

    res = 0
    for i, index in enumerate(indexes):
        res += y[i, index]
    
    return res

