import math
import torch 
from torch import nn 
import numpy as np 
from torch.nn import functional as F
from utils.util import transpose_qkv, transpose_output
from timm.models.layers import DropPath, Mlp


class Encoder(nn.Module):
    
    def __init__(self, infeature,embsize, num_hidden, head, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        self.emb = nn.Embedding(infeature, embsize)
        self.ffn1 = ffnlayer(embsize, num_hidden, head)
        self.ffn2 = ffnlayer(embsize, num_hidden, head)
        
    def forward(self, x):
        x = x + self.emb.weight
        x = self.ffn1(x)
        return self.ffn2(x)

class ffnlayer(nn.Module):
    def __init__(self, infeature, numhidden, head,*args, **kwargs) -> None:
        super(ffnlayer, self).__init__(*args, **kwargs)
        self.MultiAtt = PostiveMultiheadAttention(infeature, infeature, infeature, numhidden, head, bias=True)
        self.mlp = Mlp(numhidden, 256, numhidden, act_layer=nn.GELU, drop=0.1)
        self.ln1 = nn.LayerNorm(infeature)
    
    def forward(self, x):
        x = x + self.ln1(self.MultiAtt(x, x, x, 0.5))
        x = x + self.ln1(self.mlp(x))

        return x

class PostiveMultiheadAttention(nn.Module):
    def __init__(self, querysize, keysize, valuesize, num_hidden, heads, bias =True,*args, **kwargs) -> None:
        super(PostiveMultiheadAttention, self).__init__(*args, **kwargs)

        """
        querysize : 查询大小
        keysize   : 键大小
        valuesize : 值大小
        numhidden : 隐藏层大小
        """
        self.head = heads

        self.W_q = nn.Linear(querysize, num_hidden, bias=bias)
        self.W_k = nn.Linear(keysize, num_hidden, bias=bias)
        self.W_v = nn.Linear(valuesize, num_hidden, bias=bias)

        self.W_o = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.dropout = nn.Dropout(0.2)

    def forward(self, query, key, value, ratio):
        _, c, _ = query.shape
        q = transpose_qkv(self.W_q(query),self.head)
        k = transpose_qkv(self.W_k(key),self.head)
        v = transpose_qkv(self.W_v(value), self.head)
        # print(q.shape, k.shape, v.shape)
        out = torch.matmul(q, k.transpose(1,2))/ math.sqrt(c)
        res = F.softmax(out, dim=-1)

        res = torch.where(F.sigmoid(out) > ratio, res, 0)

        res = torch.matmul(res, self.dropout(v))
        res = self.W_o(transpose_output(res, self.head))
        # print(res.shape)

        return res




