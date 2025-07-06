import torch
from torch import nn
from torch.nn import functional as F
import numpy
from network.DFnet import DFnetBase, DFnetcls # UDA-WF
from loss import crossEntropy
from tqdm import tqdm
from d2l import torch as d2l
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
from utils.datasetLoader import DatastLoader
from utils.util import accurate
import os
from utils.dataLoaderTotal import dataLoaderSource
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier, BaggingClassifier,StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score 
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed(114514)

def load_data(feature, label): #数据生成
    '''
    dataset generator
    '''
    target = torch.tensor(feature, dtype=torch.float32).reshape(-1, 256)
    label = torch.tensor(label, dtype=torch.float32)
    targetDataset =  data.TensorDataset(target, label)

    return data.DataLoader(targetDataset, shuffle=True, batch_size=128, drop_last=True)

def kNN_train(X_train, y_train,params,n_num = 1):
    print('kNN training data shape: ', X_train.shape)
    knn = KNeighborsClassifier(n_neighbors=n_num,algorithm='auto',weights='distance')
    knn.fit(X_train, y_train)

    return knn



def kNN_accuracy(X_train, y_train, X_test, y_test, params):
    acc=[]
    auc=[]
    
    for i in range(1,10):
        knnModel = kNN_train(X_train, y_train, params,i)
        acc_knn = accuracy_score(y_test, knnModel.predict(X_test))
        print(acc_knn)
        roauc = roc_auc_score(y_test,knnModel.predict(X_test),average='macro')
        acc_knn = float("{0:.15f}".format(round(acc_knn, 6)))
        acc.append(acc_knn)
        auc.append(roauc)
    return acc, auc

def train_epoch(train_iter, test_iter, targetBase,params):
    features_traget = []
    y_t = []
    features_test = []
    y_test = []
    loop = tqdm(train_iter, desc="target")

    for X, y in loop:
        y_t.append(y.numpy())
        X, y = X.to(device), y.to(device)
        f = targetBase(X)
        features_traget.append(f.to('cpu').detach().numpy())
    
    loop = tqdm(test_iter, desc="test")

    for X, y in loop:
        y_test.append(y.numpy())
        X, y = X.to(device), y.to(device)
        f = targetBase(X)
        features_test.append(f.to('cpu').detach().numpy())
    
    features_traget = np.vstack(features_traget)
    features_test = np.vstack(features_test)
    y_t = np.vstack(y_t)
    y_test = np.vstack(y_test)
    return kNN_accuracy(features_traget, y_t, features_test, y_test, params)



def fintuning(target_iter, targetModel, clsLayer, loss, optimizer, params): 
    accmulator = d2l.Accumulator(4)
    loop = tqdm(target_iter, desc="traintgt")
    for X_t, y in loop:
        X_t, y = X_t.to(params["device"]), y.to(params["device"])
        f_t = targetModel(X_t)

        disLabel = clsLayer(f_t)
        # print(y.shape)
        # label = torch.ones(disLabel.shape[0], dtype=torch.long).to(params["device"])

        optimizer.zero_grad()
        l_tgt = loss(disLabel, y)
        l_tgt.backward()
        optimizer.step()
        accmulator.add((l_tgt) * X_t.shape[0], X_t.shape[0], accurate(disLabel, y) , y.shape[0])

        loop.set_postfix(loss = accmulator[0]/ accmulator[1], acc = accmulator[2]/ accmulator[3])

def test(test_iter, targetModel, clsLayer, loss):
    accmulator = d2l.Accumulator(4)
    loop = tqdm(test_iter, desc="test")
    labels = []
    preds = []
    for X_t, y in loop:
        X_t, y = X_t.to(params["device"]), y.to(params["device"])
        # label = torch.ones(disLabel.shape[0], dtype=torch.long).to(params["device"])
        # print(y.shape)
        with torch.no_grad():
            f_t = targetModel(X_t)
            disLabel = clsLayer(f_t)
            l_tgt = loss(disLabel, y)
            accmulator.add((l_tgt) * X_t.shape[0], X_t.shape[0], accurate(disLabel, y) , y.shape[0])
            labels.append(y.detach().cpu().numpy())
            preds.append(disLabel.detach().cpu().numpy())
        
        loop.set_postfix(loss = accmulator[0]/ accmulator[1], acc = accmulator[2]/ accmulator[3])
    print(accmulator[2]/accmulator[3] * 100)
    labels = np.vstack(labels)
    preds = np.vstack(preds)
    roauc = roc_auc_score(labels,preds,average='macro')
    return accmulator[2]/accmulator[3], roauc



    

def train(params, train_iter, test_iter, lr):

    targetModel = DFnetBase(1).to(params['device'])

    targetModel.load_state_dict(torch.load(params["target_model"]))
    clsLayer = nn.Sequential(
        nn.Linear(256, params["numclass"], bias= False)
    )
    clsLayer.to(params['device'])
    clsLayer.apply(weights_init)
    param_group = []
    for k, v in targetModel.named_parameters():
        # v.requires_grad = False
        param_group.append({'params':v, 'lr':lr})
    for k, v in clsLayer.named_parameters():
        # v.requires_grad = False
        param_group.append({'params':v, 'lr':lr})


    optimizer = torch.optim.Adam(param_group)
    loss = crossEntropy.CrossEntropy()
    best_acc = 0 
    best_auc = 0
    for i in range(params['epoch']):
        fintuning(train_iter, targetModel, clsLayer, loss, optimizer, params)
        if i % 5 == 0:
            acc, auc = test(test_iter, targetModel, clsLayer, loss)
            index = np.argmax(acc)
            if best_acc <= acc:
                best_acc = acc
                best_auc = auc
                print(best_acc, best_auc)
                torch.save(targetModel.state_dict(), params['saveBasepath'])
                torch.save(clsLayer.state_dict(), params['saveClspath'])
    
    
    targetModel = DFnetBase(1).to(params['device'])
    targetModel.load_state_dict(torch.load(params["saveBasepath"]))
    clsLayer = nn.Sequential(
        nn.Linear(256, params["numclass"], bias= False)
    )

    clsLayer.to(params['device'])
    clsLayer.apply(weights_init)
    clsLayer.load_state_dict(torch.load(params['saveClspath']))
    param_group = []
    for k, v in targetModel.named_parameters():
        param_group.append({'params':v, 'lr':lr*0.1})
    for k, v in clsLayer.named_parameters():
        param_group.append({'params':v, 'lr':lr*0.01})


    optimizer = torch.optim.Adam(param_group)
    loss = crossEntropy.CrossEntropy()
    for i in range(30):
        fintuning(train_iter, targetModel, clsLayer, loss, optimizer, params)
        if i % 5 == 0:
            acc, auc = test(test_iter, targetModel, clsLayer, loss)
            index = np.argmax(acc)
            if best_acc <= acc:
                best_acc = acc
                best_auc = auc
                print(best_acc, best_auc)
                torch.save(targetModel.state_dict(), params['saveBasepath'])
                torch.save(clsLayer.state_dict(), params['saveClspath'])
    return best_acc, best_auc
    
    

def weights_init(m):
    if type(m) == nn.Linear:
        # nn.init.kaiming_uniform_(m.weight)
        # nn.init.kaiming_normal_(m.weight)
        nn.init.xavier_uniform_(m.weight)


params = {}

device = d2l.try_gpu()
params['device'] = device

# dataset setting
params['target_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/target_processed/WF15.npz"
params['test_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/target_test/WF_test.npz"
train_iter = dataLoaderSource(params["target_data"], 128)
test_iter = dataLoaderSource(params["test_data"], 128)


#train setting
params["target_model"] = "XXX.pth"
params["saveBasepath"] = "targetBestModelBase.pth"
params["saveClspath"] = "targetBestModelCls.pth"
params['epoch'] = 80
params["numclass"] = 90

acc_list = []
auc_list = []
total = 0
for i in range(0, 1):
    acc, auc = train(params, train_iter, test_iter, lr=0.001)
    acc_list.append(acc)
    auc_list.append(auc)    
    total +=acc
    print(acc_list,auc_list)
print(total/3)
print(np.array(auc_list).mean())

