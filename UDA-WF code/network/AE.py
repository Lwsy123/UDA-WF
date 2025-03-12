import torch 
from torch import nn
import numpy as np 

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = nn.Sequential(
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )
    def forward(self, X):
        return self.block1(X)

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = nn.Sequential(
        nn.LazyLinear(128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.LazyLinear(256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        )
    def forward(self, X):
        return self.block1(X)

class AutoEncoder(nn.Module):
    def __init__(self,latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.latent = nn.LazyLinear(latent_dim,bias=False)
        self.ln = nn.LazyLinear(128, bias=False)
        self.decoder = Decoder()
    
    def forward(self, X):
        X = self.encoder(X)
        l = self.latent(X)
        X = self.ln(l)
        return l, self.decoder(X)