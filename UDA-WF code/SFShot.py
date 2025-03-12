import torch 
from torch import nn
from torch.nn import functional as F
from loss import weightedCrossEntropy, IMLoss,IMLoss_SHOT
from tqdm import tqdm
from network.DFnet import DFnetBase, DFnetcls
# from network.newDFnet import DFnetBase, DFnetcls
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

def load_data(params): #数据生成
    '''
    返回值: 数据生成器, 原始数据
    '''
    # sourcepath = params["source_data"]
    targetpath = params["target_data"]

    # source_data = np.load(sourcepath)
    target_data = np.load(targetpath)

    # source = source_data['x'].reshape(-1, 1, 5000)
    target = target_data['x'].reshape(-1, 1, 5000)

    # source = DatastLoader(source)
    targetDataset = DatastLoader(target) 

    return data.DataLoader(targetDataset, shuffle=False, batch_size=128), target


def train_epoch(epochname, train_iter, netBase, netCls, optimizer, loss_IM, loss_CE, device):
    # 生成训练

    loop = tqdm(train_iter, desc=epochname)
    accumulator = d2l.Accumulator(4)

    for X, y, label_weight in loop:

        X, y, label_weight = X.to(device), y.to(device), label_weight.to(device)

        y_pred = netCls(netBase(X))

        acc = accurate(y_pred, y)

        optimizer.zero_grad()

        l = loss_IM(y_pred, 100)
        # print(l)
        # l = 0
        l += loss_CE(y_pred, y, label_weight)
        l.backward()
        optimizer.step()

        accumulator.add(l * y.numel(), y.numel(), acc, y.shape[0])

        loop.set_postfix(loss = accumulator[0]/accumulator[1], acc=accumulator[2]/accumulator[3])

def train(train_iter, dataset,  lr, params):

    # 加载迁移的模型参数,用源域参数进行初始化
    netBase = DFnetBase(1) 
    netCls = DFnetcls(100)
    netBase.load_state_dict(torch.load(params["sourcemodel"]))
    netCls.load_state_dict(torch.load(params["sourcecls"]))
    # netCls = nn.Sequential(
    #     nn.LazyLinear(100,bias=False)
    # )
    # 将模型导入GPU
    netBase.to(device)
    netCls.to(device)

    param_group = []
    for k, v in netBase.named_parameters(): # 获取netBase的参数
        param_group.append({'params':v, 'lr':lr})

    for k, v in netCls.named_parameters(): # 获取netCls的参数
        v.requires_grad = False
    
    optimizer = torch.optim.Adam(param_group) #使用Adam优化器
    '''
    SDA-WF
    '''
    loss_IM =  IMLoss.IMLoss()# 平滑交叉熵
    '''
    SHOT
    '''
    # loss_IM = IMLoss_SHOT.IMLossSHOT()
    loss_CE = weightedCrossEntropy.WeightedCrossEntropy()
    # 将模型导入GPU
    netBase.to(device)
    netCls.to(device)
    
    for i in range(params["epoch"]):
        epochname = 'epoch' + str(i)

        # netBase.train()
        # netCls.eval()

        if i % params["maxiter"] == 0: # 每maxiter 轮生成一次伪标签
            '''
            SDA-WF
            '''
            label, label_weight = obtain_label(train_iter, netBase, netCls, device) # 生成伪标签以及伪标签的权重
            label_weight = label_weight.reshape(-1, 1)
            print(label_weight.reshape(-1).min())
            '''
            SHOT
            '''
            # label,_ = obtain_label_shot(train_iter, netBase, netCls, device)
            # print(label.shape)
            # label_weight = np.ones(label.shape[0], dtype=np.float32).reshape(-1, 1)
            # print(np.unique(np.argwhere(label ==1)[:, 1], return_counts=True))
            dataset_afoda = dataLoaderAfterODA(dataset.reshape(-1, 5000), label, label_weight, 128) #生成数据迭代器
        train_epoch(epochname, dataset_afoda , netBase, netCls, optimizer, loss_IM, loss_CE, device) #参数训练

        # acc = test_epoch(test_iter, netBase, netCls, loss, device)

        torch.save(netBase.state_dict(), params["savepath"]) #保存目标域数据特征提取模型的参数

device = d2l.try_gpu()

# 目标域数据集
params = {}

device = d2l.try_gpu()
params['device'] = device

# 数据集地址
# params['target_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/AWF/AWF5.npz"
params['target_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/target_processed/WF15.npz"
# params['target_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/WangDataset/Wang100_sampled20.npz"
# params['target_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/DFDataset/DF95_sampled20.npz"
# params['target_data'] = "/home/siyuwang/pythoncode/TranferLearning/Dataset/AFDataset/AF100_sampled20.npz"
'''
SDA-WF
'''
train_iter, dataset = load_data(params)
'''
SHOT
''' 
# dataset = np.load(params["target_data"])['x']
# train_iter = dataLoaderSource(params["target_data"], 128)


#训练
params["sourcemodel"] = "/home/siyuwang/pythoncode/TranferLearning/Best_model/souce_train/bestBaseSE25.pth"
params["sourcecls"] = "/home/siyuwang/pythoncode/TranferLearning/Best_model/souce_train/bestClsSE25.pth"

params["savepath"] = "targetBestSEModel.pth"
params['epoch'] = 70

params["maxiter"] = 10
start_time = time.time()
train(train_iter, dataset, 0.0001, params)
end_time = time.time()

print(end_time - start_time)

