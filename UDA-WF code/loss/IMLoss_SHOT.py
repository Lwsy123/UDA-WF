import torch 
from torch import nn 
from torch.nn import functional as F

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=-1)
    return entropy 

class IMLossSHOT(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(IMLossSHOT, self).__init__(*args, **kwargs)

    def forward(self, y_pred):
        softmax_out = F.softmax(y_pred, dim = -1)

        l_ent = torch.mean(Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0) #根据行求平均
        l_div = torch.sum(- msoftmax  * torch.log(msoftmax + 1e-6))

        im_loss =  l_ent - l_div
        # im_loss = l_ent

        return im_loss
 
