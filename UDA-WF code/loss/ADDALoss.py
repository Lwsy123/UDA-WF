import torch
from torch import nn
from torch.nn import functional as F


class ADDALoss(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(ADDALoss, self).__init__(*args, **kwargs)

    def forward(self, source, target):
        source_pos = source[:, 0]
        target_nagtive = target[:, 1]
        target_pos = target[:, 0]

        return -torch.log(source_pos).mean() - torch.log(target_pos).mean() - torch.log(target_nagtive).mean()
        
