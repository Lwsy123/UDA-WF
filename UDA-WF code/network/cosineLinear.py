import torch
from torch import nn

class CosineLinear(nn.Module):
    def __init__(self, output, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ln = nn.Linear(256, output, bias=False)
    
    def forward(self, x):
        norm_x = x/ torch.norm(x, p=2, dim=0, keepdim=True)
        wstar = self.ln.weight.T
        norm_Wstar = wstar / torch.norm(wstar, p=2, dim=0, keepdim=True)
        return torch.mm(norm_x, norm_Wstar)