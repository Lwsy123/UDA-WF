import torch 
from torch import nn 
from torch.nn import functional as F 
import numpy as np
from .AttentionEncoder import Encoder

filter_num = [32, 64, 128, 256]


class DFnetBlock(nn.Module):

    def __init__(self, infeature, outfeature, stride=1,padding='same', use_ElU = False, *args, **kwargs) -> None:
        super(DFnetBlock, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv1d(infeature, outfeature, kernel_size=8, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(outfeature, outfeature, kernel_size=8, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=4, padding= 4)
        self.dropout = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(outfeature)

        if use_ElU:
            self.activation = nn.ELU()
        else :
            self.activation = nn.ReLU()

    def forward(self, X):

        # seq1
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.activation(X)
        X = self.dropout(X)

        # seq2
        X = self.conv2(X)
        X = self.bn1(X)
        X = self.activation(X)
        X = self.dropout(X)

        X = self.maxpool(X)
        X = self.dropout(X)

        return X

class DFnetBase(nn.Module):

    def __init__(self, infeature,*args, **kwargs) -> None:
        super(DFnetBase, self).__init__(*args, **kwargs)

        self.seq1 = DFnetBlock(infeature, filter_num[0], use_ElU=True)
        self.seq2 = DFnetBlock(filter_num[0], filter_num[1])
        self.seq3 = DFnetBlock(filter_num[1], filter_num[2])
        self.seq4 = DFnetBlock(filter_num[2], filter_num[3])
        self.proj = nn.Conv1d(20, 1, kernel_size=1)
        self.flattent = nn.Flatten()
        self.gmax = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, X):

        X = self.seq1(X)
        X = self.seq2(X)
        X = self.seq3(X)
        X = self.seq4(X)
        X =  X.mean(dim=-1)
        return X


class DFnetcls(nn.Module):
    def __init__(self, num_cls, *args, **kwargs) -> None:
        super(DFnetcls, self).__init__(*args, **kwargs)
        self.ln1 = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
        )
        self.cls = nn.LazyLinear(num_cls)

    def forward(self, x):

        # x = self.dfbase(x)
        # x = self.ln1(x)
        x = self.cls(x)
        
        return x
