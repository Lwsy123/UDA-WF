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

# class AutoEncoder(nn.Module):
#     def __init__(self,latent_dim, out_dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.encoder = nn.Sequential(nn.LazyLinear(latent_dim),
                                     
#                                      )
#         self.decoder = nn.LazyLinear(out_dim)

class DFnetBase(nn.Module):

    def __init__(self, infeature,*args, **kwargs) -> None:
        super(DFnetBase, self).__init__(*args, **kwargs)

        self.seq1 = DFnetBlock(infeature, filter_num[0], use_ElU=True)
        # self.SE1 = SE_Block(filter_num[0],8)
        self.seq2 = DFnetBlock(filter_num[0], filter_num[1])
        # self.SE2 = SE_Block(filter_num[1],8)
        self.seq3 = DFnetBlock(filter_num[1], filter_num[2])
        # self.SE3 = SE_Block(filter_num[2], 8)
        self.seq4 = DFnetBlock(filter_num[2], filter_num[3])
        # self.SE4 = SE_Block(filter_num[3],8)
        self.proj = nn.Conv1d(20, 1, kernel_size=1)
        self.flattent = nn.Flatten()
        self.gmax = nn.AdaptiveMaxPool1d(1)
        # self.proj = nn.Conv1d(256, 256, kernel_size=1)
        # self.proj = nn.Sequential(
        #     nn.Linear(256),
        #     nn.Linear(128),
        #     nn.Linear(256)

        # )
        # self.ln1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.LazyLinear(512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.7),
        #     nn.LazyLinear(512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.7),
        # )
        # self.encoder = Encoder.Encoder(20, filter_num[3], filter_num[3], 8)
    
    def forward(self, X):

        X = self.seq1(X)
        # X = self.SE1(X)
        X = self.seq2(X)
        # X = self.SE2(X)
        X = self.seq3(X)
        # X = self.SE3(X)
        X = self.seq4(X)
        # Z = X.transpose(1,2)
        # Z = self.proj(Z).reshape(-1,256)

        # Y = torch.matmul(X, X.transpose(1,2))/torch.sqrt(torch.tensor(X.shape[1]))
        # Y = F.softmax(Y)
        # X = torch.matmul(Y, X)
        # print(X.shape)

        
        # X = self.SE4(X)
        # X = self.flattent(X)
        # X = self.ln1(X)
        # X = X.transpose(1,2)
        # X = self.encoder(X)
        # X = self.gmax(X).reshape(-1, X.shape[1])#X.max(dim=-1)
        # print(X.shape)
        X =  X.mean(dim=-1)

        # X = X * F.softmax(self.proj(X.reshape(-1, 256, 1)).reshape(-1, 256))

        return X


class DFnetcls(nn.Module):
    def __init__(self, num_cls, *args, **kwargs) -> None:
        super(DFnetcls, self).__init__(*args, **kwargs)

        #self.dfbase = DFnetBase(infeature)
        self.ln1 = nn.Sequential(
            # nn.Flatten(),
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


#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1)
            # weight = F.sigmoid(y)
            # y = torch.where(y >= 0.7, y, 0)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)

