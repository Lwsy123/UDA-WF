import torch 
from torch import nn
from torch.nn import functional as F
from loss import weightedCrossEntropy, IMLoss,IMLoss_SHOT
from tqdm import tqdm
from network.DFnet import DFnetBase, DFnetcls
from d2l import torch as d2l
from utils.SHOT import obtain_label_shot
from utils.util import accurate
import os
from utils.pGenLabel import obtain_label
from utils.dataLoaderTotal import  dataLoaderAfterODA, dataLoaderSource
import numpy as np
from utils.datasetLoader import DatastLoader
from torch.utils import data
import time
# torch.cuda.manual_seed(1024)

def load_data(params):
    '''
    dataset generator
    '''
    targetpath = params["target_data"]

    target_data = np.load(targetpath)

    target = target_data['x'].reshape(-1, 1, 5000)
    targetDataset = DatastLoader(target) 

    return data.DataLoader(targetDataset, shuffle=False, batch_size=128), target


def train_epoch(epochname, train_iter, netBase, netCls, optimizer, loss_IM, loss_CE, device):

    loop = tqdm(train_iter, desc=epochname)
    accumulator = d2l.Accumulator(4)

    for X, y, label_weight in loop:

        X, y, label_weight = X.to(device), y.to(device), label_weight.to(device)

        y_pred = netCls(netBase(X))

        acc = accurate(y_pred, y)

        optimizer.zero_grad()

        l = loss_IM(y_pred, 100)
        l += loss_CE(y_pred, y, label_weight)
        l.backward()
        optimizer.step()

        accumulator.add(l * y.numel(), y.numel(), acc, y.shape[0])

        loop.set_postfix(loss = accumulator[0]/accumulator[1], acc=accumulator[2]/accumulator[3])

def train(train_iter, dataset,  lr, params):

    netBase = DFnetBase(1) 
    netCls = DFnetcls(100)
    netBase.load_state_dict(torch.load(params["sourcemodel"]))
    netCls.load_state_dict(torch.load(params["sourcecls"]))
    netBase.to(device)
    netCls.to(device)

    param_group = []
    for k, v in netBase.named_parameters():
        param_group.append({'params':v, 'lr':lr})

    for k, v in netCls.named_parameters(): 
        v.requires_grad = False
    
    optimizer = torch.optim.Adam(param_group)
    '''
    UDA-WF
    '''
    loss_IM =  IMLoss.IMLoss()
    loss_CE = weightedCrossEntropy.WeightedCrossEntropy()
    netBase.to(device)
    netCls.to(device)
    
    for i in range(params["epoch"]):
        epochname = 'epoch' + str(i)

        if i % params["maxiter"] == 0: 
            '''
            SDA-WF
            '''
            label, label_weight = obtain_label(train_iter, netBase, netCls, device) # generate pseudo-label and label-weight
            label_weight = label_weight.reshape(-1, 1)
            print(label_weight.reshape(-1).min())
            dataset_afoda = dataLoaderAfterODA(dataset.reshape(-1, 5000), label, label_weight, 128) 
        train_epoch(epochname, dataset_afoda , netBase, netCls, optimizer, loss_IM, loss_CE, device) 

        torch.save(netBase.state_dict(), params["savepath"])

device = d2l.try_gpu()

params = {}

device = d2l.try_gpu()
params['device'] = device


params['target_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/target_processed/WF15.npz"
'''
UDA-WF
'''
train_iter, dataset = load_data(params)

params["sourcemodel"] = "/home/siyuwang/pythoncode/TranferLearning/Best_model/souce_train/bestBaseSE25.pth"
params["sourcecls"] = "/home/siyuwang/pythoncode/TranferLearning/Best_model/souce_train/bestClsSE25.pth"

params["savepath"] = "targetBestSEModel.pth"
params['epoch'] = 70

params["maxiter"] = 10
start_time = time.time()
train(train_iter, dataset, 0.0001, params)
end_time = time.time()

print(end_time - start_time)

