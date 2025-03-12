import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np

path = "/home/siyuwang/pythoncode/TranferLearning/openDataset/LargeOpen/AWF_sampled2000.npz"
labledeck = dict()

dataset = np.load(path, allow_pickle=True)

dataset_x = dataset['x']
dataset_Y = dataset['y']
labels = np.eye(101) # opendataset Label
print(labels.shape)

index = np.arange(dataset_Y.shape[0]) 
dataset_y = []
for i in range(dataset_Y.shape[0]):
    dataset_y.append(100)

# index_select = np.random.choice(index, 7000, replace=False) # 选择测试集
# # print(dataset_y[index_select])
# WF_x_test = dataset_x[index_select]
# WF_y_test = labels[np.ones(7000,dtype=np.int32) *100]
# print(WF_y_test[0])

# np.savez("./openDataset/target_opentest/WF_opentest", x = WF_x_test, y = WF_y_test)
# index_select = np.array(index_select).reshape(-1) 
# index = np.arange(0 , dataset_x.shape[0])
# index = np.setdiff1d(index, index_select) # 选择训练集

# dataset_x = dataset_x[index] 
for i in [500, 1000, 1500, 2000]:
    index_select = np.random.choice(np.arange(dataset_x.shape[0]), i, replace=False)
    WF_x = dataset_x[index_select]
    WF_y = labels[np.ones(i,dtype=np.int32) *100]
    np.savez("./openDataset/AWFopenDataset/AWFopen" + str(i), x = WF_x, y = WF_y)
