from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import torch
from torch import nn
from torch.nn import functional as F
import numpy
from network.DFnet import DFnetBase, DFnetcls
from loss import ADDALoss, crossEntropy
from tqdm import tqdm
from d2l import torch as d2l
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
from utils.datasetLoader import DatastLoader
from utils.util import accurate
import os
from utils.dataLoaderopenTotal import dataLoaderSource
from utils.dataLoaderTotal import dataLoadercloseSource
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.evaluation import evluation
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
import warnings
from xgboost.sklearn import XGBClassifier
warnings.filterwarnings("ignore")

def load_data(feature, label): #dataset generator
    '''
    dataset generator
    '''
    target = torch.tensor(feature, dtype=torch.float32).reshape(-1, 256)
    label = torch.tensor(label, dtype=torch.float32)
    targetDataset =  data.TensorDataset(target, label)

    return data.DataLoader(targetDataset, shuffle=True, batch_size=128, drop_last=True)

def fintuning(target_iter, targetModel, clsLayer, taskLayer, loss, optimizer, params): 
    accmulator = d2l.Accumulator(4)
    loop = tqdm(target_iter, desc="traintgt")
    for X_t, y, openlabel in loop:
        X_t, y, openlabel = X_t.to(params["device"]), y.to(params["device"]), openlabel.to(params['device'])
        f_t = targetModel(X_t)

        disLabel = clsLayer(f_t)

        optimizer.zero_grad()
        l_tgt = loss(disLabel, y)
        l_tgt.backward()
        optimizer.step()
        accmulator.add((l_tgt) * X_t.shape[0], X_t.shape[0], accurate(disLabel, y) , y.shape[0])

        loop.set_postfix(loss = accmulator[0]/ accmulator[1], acc = accmulator[2]/ accmulator[3])

def test(test_iter, targetModel, clsLayer, loss):
    accmulator = d2l.Accumulator(4)
    loop = tqdm(test_iter, desc="test")
    label = []
    y_pred = []
    for X_t, y, openlabel in loop:
        X_t, y = X_t.to(params["device"]), y.to(params["device"])
        with torch.no_grad():
            f_t = targetModel(X_t)
            disLabel = clsLayer(f_t)
            label.append(y.cpu().numpy())
            y_pred.append(disLabel.cpu().numpy())
            l_tgt = loss(disLabel, y)
            accmulator.add((l_tgt) * X_t.shape[0], X_t.shape[0], accurate(disLabel, y) , y.shape[0])

        loop.set_postfix(loss = accmulator[0]/ accmulator[1], acc = accmulator[2]/ accmulator[3])
    label = np.concatenate(label, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    print(label.shape, y_pred.shape)
    return evluation(y_pred, label), roc_auc_score(label,y_pred,average='macro')

    

def train(params, train_iter_close, train_iter, test_iter, lr):

    targetModel = DFnetBase(1).to(params['device'])

    targetModel.load_state_dict(torch.load(params["target_model"]))
    clsLayer = nn.Sequential(
        nn.LazyLinear(params["numclass"],bias=False)
    )
    taskLayer = nn.Sequential(
        nn.LazyLinear(2, bias=False)
    )
    clsLayer.to(params['device'])
    clsLayer.apply(weights_init)
    taskLayer.to(params["device"])
    param_group = []
    for k, v in targetModel.named_parameters():
        param_group.append({'params':v, 'lr':lr})
    for k, v in clsLayer.named_parameters():
        param_group.append({'params':v, 'lr':lr })


    optimizer = torch.optim.Adam(param_group)
    loss = crossEntropy.CrossEntropy()
    best_f1 = 0
    best_auc = 0
    best_eval = None
    for i in range(params['epoch']):
        fintuning(train_iter, targetModel, clsLayer,taskLayer, loss, optimizer, params)
        
        if i % 5 == 0:
            eval_score, auc = test(test_iter, targetModel, clsLayer, loss)
            print(eval_score, auc)
            if eval_score[2] >= best_f1:
                best_f1 = eval_score[2]
                best_eval = eval_score
                best_auc = auc
                torch.save(targetModel.state_dict(), params['saveBasepath'])
                torch.save(clsLayer.state_dict(), params['saveClspath'])
            print(best_eval)
            print(best_auc)
    
    
    targetModel = DFnetBase(1).to(params['device'])

    targetModel.load_state_dict(torch.load(params["saveBasepath"]))
    clsLayer = nn.Sequential(
        nn.LazyLinear(params["numclass"],bias=False)
    )
    clsLayer.load_state_dict(torch.load(params["saveClspath"]))
    taskLayer = nn.Sequential(
        nn.LazyLinear(2, bias=False)
    )
    param_group = []
    for k, v in targetModel.named_parameters():
        param_group.append({'params':v, 'lr':lr*0.1})
    for k, v in clsLayer.named_parameters():
        param_group.append({'params':v, 'lr':lr*0.01 })


    optimizer = torch.optim.Adam(param_group)
    loss = crossEntropy.CrossEntropy()
    for i in range(0):
        fintuning(train_iter, targetModel, clsLayer,taskLayer, loss, optimizer, params)
        
        if i % 5 == 0:
            eval_score, auc = test(test_iter, targetModel, clsLayer, loss)
            if eval_score[2] >= best_f1:
                best_f1 = eval_score[2]
                best_eval = eval_score
                best_auc = auc
                torch.save(targetModel.state_dict(), params['saveBasepath'])
                torch.save(clsLayer.state_dict(), params['saveClspath'])
            print(best_eval)
            print(best_auc)
    
  
    return best_eval, best_auc
    

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


params = {}

device = d2l.try_gpu()
params['device'] = device

'''
closedataset
'''
params['target_close_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/DFDataset/WF90_sampled5.npz"
'''
opendataset
'''
params['target_open_data'] = "/home/siyuwang/pythoncode/TranferLearning/openDataset/DFopenDataset/WF_sampled500.npz"

'''
testclose
'''
params['test_close_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/target_test/WF90_sampled70.npz"
'''
testopen
'''
params['test_open_data'] = "/home/siyuwang/pythoncode/TranferLearning/openDataset/target_opentest/WF_sampled7000.npz"

train_iter = dataLoaderSource(params["target_close_data"],params['target_open_data'], 128)
train_iter_close = dataLoadercloseSource(params["target_close_data"], 128)
test_iter = dataLoaderSource(params["test_close_data"], params['test_open_data'],128)


#train setting
params["target_model"] = "/home/siyuwang/pythoncode/TranferLearning/targetBestSEModel.pth"
params["saveBasepath"] = "targetBestModelBase_open.pth"
params['saveClspath'] = "cls_open.pth"

params['epoch'] = 80
params['numclass'] = 91
acc_list = []
auc_list = []
for i in range(0, 1):
    acc, auc = train(params, train_iter_close, train_iter, test_iter, lr=0.001)
    acc_list.append(acc)
    auc_list.append(auc)
    print(acc_list, auc_list)
acc_list = np.array(acc_list)
auc_list = np.array(auc_list)
print(acc_list.mean(axis=0))
print(auc_list.mean())
