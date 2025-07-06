import time
import torch 
from torch import nn
from torch.nn import functional as F
from loss.smoothCrossEntropy import smoothCrossEntropy
from tqdm import tqdm
from network.DFnet import DFnetBase, DFnetcls
from d2l import torch as d2l
from utils.util import accurate
import os
from utils.dataLoaderTotal import dataLoaderSource
import numpy as np

def train_epoch(epochname, train_iter, netBase, netCls, optimizer, loss, device):
    
    loop = tqdm(train_iter, desc=epochname)
    accumulator = d2l.Accumulator(4)

    for X, y in loop:

        X, y = X.to(device), y.to(device)

        y_pred = netCls(netBase(X))

        acc = accurate(y_pred, y)

        optimizer.zero_grad()
        l = loss(y_pred, y, 0.1)
        l.backward()
        optimizer.step()

        accumulator.add(l * y.numel(), y.numel(), acc, y.shape[0])

        loop.set_postfix(totalLoss= accumulator[0]/accumulator[1], acc=accumulator[2]/accumulator[3])


def test_epoch(test_iter, netBase, netCls, loss, device):
    loop = tqdm(test_iter, desc='testepoch')
    accumulator = d2l.Accumulator(4)
    netBase.eval()
    netCls.eval()

    feature = []
    labels = []
    for X, y in loop:
         
        with torch.no_grad():
            X, y = X.to(device), y.to(device)

            y_pred = netBase(X)
        feature.append(y_pred.cpu().numpy())
        labels.append(y.cpu().numpy())

    feature = np.vstack(feature)
    labels = np.vstack(labels)
    print(feature.shape, labels.shape)

    np.savez("./source_features", feature = feature, label = labels)
        
        


def train(train_iter, epochnum, lr, device, num):

    netBase = DFnetBase(1)
    netCls = DFnetcls(100)

    param_group = []
    for k, v in netBase.named_parameters(): # parameters of netBase
        param_group.append({'params':v, 'lr':lr})

    for k, v in netCls.named_parameters(): # parameters of netCls
        param_group.append({'params':v, 'lr':lr})
    
    max_acc = 0
    optimizer = torch.optim.Adam(param_group) 
    loss = smoothCrossEntropy()
    
    # 将模型导入GPU
    netBase.to(device)
    netCls.to(device)

    netBase.apply(weights_init)
    netCls.apply(weights_init)
    max_acc = 0
    path = "./Best_model/souce_train/"
    pathBase = path + "bestBaseSE" + str(num) + ".pth"
    pathCls = path + "bestClsSE" + str(num) + ".pth"
    
    for i in range(epochnum):
        epochname = 'epoch' + str(i)

        netBase.eval()
        netCls.eval()
        train_epoch(epochname, train_iter, netBase, netCls, optimizer, loss, device)
        torch.save(netBase.state_dict(), pathBase)
        torch.save(netCls.state_dict(), pathCls)

def weights_init(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

device = d2l.try_gpu()
for j in [25]:
    start_time = time.time()
    train_iter = dataLoaderSource("./Dataset/processed/AWF25.npz", 128)
    
    train(train_iter, 50, 0.001, device , j)
    end_time = time.time()
    print(end_time - start_time)



    
    
