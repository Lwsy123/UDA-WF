import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
import math

    
def obtain_label(loader, netF, netB, device):
    start_test = True
    # obtain extracted features
    with torch.no_grad():
        for X in loader:
            # data = iter_test.next()
            inputs = X.to(device)
            feas = netF(inputs)
            outputs = netB(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    # obtain predicted labels
    all_output = nn.Softmax(dim=-1)(all_output)
    _, predict = torch.max(all_output, 1)
    
    # cluster features based on cosine similarity
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1) 
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    mu = 0
    var = 1
    for _ in range(20):
        initc = aff.transpose().dot(all_fea) # clustering center
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>0)
        
        labelset = labelset[0]
        

        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=-1)
        
        predict = labelset[pred_label]
        
        # EMA
        min_dist = dd.min(axis=-1)
        mu_b = min_dist.sum()/min_dist.shape[0] # average distance
        var_b = np.power((min_dist - mu_b),2).sum()/min_dist.shape[0] # distance variance
        mu = 0.5 * mu + 0.5 * mu_b
        var = 0.5 * var + 0.5 * var_b
        aff = np.eye(K)[predict]
    GussianDis = np.exp(-np.power((min_dist-mu),2)/(2*var))
    Label_weight = np.where(min_dist <= mu, 1, GussianDis)
    print(Label_weight.min(), Label_weight.max(), mu, var)

    label = np.eye(K)[predict]
    
    return label, Label_weight
