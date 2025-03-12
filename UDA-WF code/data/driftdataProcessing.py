import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np

path = "./Dataset/tarsecondDataset.npz"
labledeck = dict()

dataset = np.load(path, allow_pickle=True)

dataset_x = dataset['x']
dataset_Y = dataset['y']
labels = np.unique(dataset_Y)
for i, y in enumerate(labels):
    labledeck[y] = i

labels = np.eye(labels.shape[0])

dataset_y = []
for y in dataset_Y:
    dataset_y.append(labledeck[y])

dataset_y = np.array(dataset_y)

WF_x_test = np.empty([0, 5000],dtype=int) # 得到空数据
WF_y_test = np.empty([0, labels.shape[0]],dtype=int)
print(labels.shape)
# selcted_index = []
# 获得测试数据集
for i in range(len(labels)):
    print(i)
    index = np.argwhere(dataset_y == i).reshape(-1)
    # index_select = np.random.choice(index, 70, replace=False)
    # selcted_index.append(index_select) #获得已被选择的测试数据
    x = dataset_x[index] # 得到当前的数据
    y = labels[np.ones(len(index), dtype=int) * i] # 得到当前的label
    WF_x_test = np.append(WF_x_test, x, axis=0) # 追加数据 
    WF_y_test = np.append(WF_y_test, y, axis=0) # 追加标签数据
print(WF_y_test.shape)
np.savez("./Dataset/target_test/WF_secondtest", x = WF_x_test, y = WF_y_test)
# index_select = np.array(index_select).reshape(-1)
# index = np.arange(0 , dataset_x.shape[0])
# index = np.setdiff1d(index, index_select)