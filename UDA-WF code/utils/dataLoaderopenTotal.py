import torch 
from torch import nn 
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import dataloader
import  numpy as np

# load softmatch data 
def dataLoaderAfterODA(tardata, label, weight,batchsize):

    dataset = np.concatenate((tardata, label, weight), axis=1)

    np.random.shuffle(dataset)

    tardata, label, weight = torch.tensor(dataset[:, :5000], dtype=torch.float32).reshape(-1, 1, 5000), torch.tensor(dataset[:, 5000:-1], dtype=torch.float32), torch.tensor(dataset[:, -1],dtype=torch.float32)

    dataset = data.TensorDataset(tardata,label, weight)

    return data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

def dataLoaderBeforeODA(dataset, batchsize):
    dataset = torch.tensor(dataset,dtype=torch.float32)
    dataset = data.TensorDataset((dataset,))

    return data.DataLoader(dataset, batch_size=batchsize)

# closeworld源域训练数据
def dataLoaderSource(closepath, openpath,batchsize):
    closedataset = np.load(closepath, allow_pickle=True)
    opendataset = np.load(openpath, allow_pickle=True)
    closedataset_x, closedataset_y = closedataset['x'],closedataset['y']
    opendataset_x, opendataset_y = opendataset['x'],opendataset['y'].astype(np.int32)
    closelabel = np.zeros(closedataset_x.shape[0], dtype=np.int)
    openlabel = np.ones(opendataset_x.shape[0], dtype=np.int)
    open_label = np.hstack((closelabel, openlabel))

    open_label = np.eye(2)[open_label]

    print(opendataset_y[0])
    dataset_x = torch.tensor(np.vstack((closedataset_x, opendataset_x)), dtype=torch.float32).reshape(-1, 1, 5000)
    dataset_y = getdataset_y(closedataset_y, opendataset_y)
    # dataset_y = torch.tensor(np.vstack((closedataset_y, opendataset_y)), dtype=torch.int)
    print(dataset_y.shape)
    if len(dataset_y.shape) != 2: 
        dataset_y = torch.eye(torch.unique(dataset_y).shape[0])[dataset_y]
    print(dataset_y.shape)
    dataset_open = torch.tensor(open_label, dtype=torch.float32)
    dataset = data.TensorDataset(dataset_x, dataset_y, dataset_open)

    return data.DataLoader(dataset, batch_size=batchsize, shuffle=True)


def getdataset_y(closedataset_y, opendataset_y):
    if len(closedataset_y.shape) == 2:
        closedataset_y = np.argwhere(closedataset_y == 1)[:,1].reshape(-1)
    labelmax = closedataset_y.max()
    
    if len(opendataset_y.shape) == 2:
        opendataset_y = np.argwhere(opendataset_y == 1)[:, 1].reshape(-1)
    
    opendataset_y = np.ones_like(opendataset_y) * (labelmax + 1)
    print(opendataset_y[0])
    dataset_y = np.concatenate((closedataset_y, opendataset_y), axis=0)
    return torch.tensor(dataset_y, dtype=torch.int)
