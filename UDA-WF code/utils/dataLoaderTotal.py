import torch 
from torch import nn 
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import dataloader
import  numpy as np

# load softmatch data 
def dataLoaderAfterODA(tardata, label, weight, batchsize, open_label = None):
    if open_label is None:
        dataset = np.concatenate((tardata, label, weight), axis=1)

        np.random.shuffle(dataset)

        tardata, label, weight = torch.tensor(dataset[:, :5000], dtype=torch.float32).reshape(-1, 1, 5000), torch.tensor(dataset[:, 5000:-1], dtype=torch.float32), torch.tensor(dataset[:, -1],dtype=torch.float32)

        dataset = data.TensorDataset(tardata,label, weight)
    else :
        open_label = np.eye(2)[open_label]
        # open_label = open_label.reshape(-1, 1)
        dataset = np.concatenate((tardata, label, weight, open_label), axis=1)

        np.random.shuffle(dataset)

        tardata, label, weight, openlabel = torch.tensor(dataset[:, :5000], dtype=torch.float32).reshape(-1, 1, 5000), torch.tensor(dataset[:, 5000:-3], dtype=torch.float32), torch.tensor(dataset[:, -3],dtype=torch.float32), torch.tensor(dataset[:, -2:], dtype=torch.float32)
        # openlabel = openlabel.reshape(-1)
        # openlabel = torch.eye(2)[openlabel].reshape(-1, 1)
        print(openlabel.shape)
        dataset = data.TensorDataset(tardata,label, weight, openlabel)


    return data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

def dataLoaderBeforeODA(dataset, batchsize):
    dataset = torch.tensor(dataset,dtype=torch.float32)
    dataset = data.TensorDataset((dataset,))

    return data.DataLoader(dataset, batch_size=batchsize)

# closeworld源域训练数据
def dataLoaderopenSource(path, batchsize):
    closedataset = np.load(path["target_close_data"])
    opendataset = np.load(path["target_open_data"])
    dataset_x  = np.vstack((closedataset['x'], opendataset['x']))
    opendataset_y = np.ones(opendataset['y'].shape[0],dtype= np.int32)*90
    closedataset_y = np.argwhere(closedataset['y'] == 1)[:,1].reshape(-1)
    print(closedataset_y.shape)
    dataset_y = np.hstack((closedataset_y, opendataset_y))
    dataset_x, dataset_y = torch.tensor(dataset_x,dtype=torch.float32).reshape(-1, 1, 5000), torch.tensor(dataset_y, dtype=torch.int)
    if len(dataset_y.shape) != 2: 
        dataset_y = torch.eye(torch.unique(dataset_y).shape[0])[dataset_y]
    print(dataset_y.shape)
    # opendataset_y = np.ones_like(opendataset['y'].shape[0],dtype= np.int32)*90
    # dataset_y = np.vstack((dataset_y, opendataset['y']))
    # opendataset_x, opendataset_y = torch.tensor(opendataset['x'],dtype=torch.float32).reshape(-1, 1, 5000), torch.tensor(opendataset['y'], dtype=torch.int)
    # dataset_y = torch.eye(K)[dataset_y]
    print(dataset_x.shape)
    # if len(dataset_y.shape) != 2: 
    #     dataset_y = torch.eye(torch.unique(dataset_y).shape[0])[dataset_y]
    # print(dataset_y.shape)
    dataset = data.TensorDataset(dataset_x, dataset_y)

    return data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

def dataLoaderSource(path, batchsize):
    dataset = np.load(path)
    dataset_x, dataset_y = torch.tensor(dataset['x'],dtype=torch.float32).reshape(-1, 1, 5000), torch.tensor(dataset['y'], dtype=torch.int)
    # dataset_y = torch.eye(K)[dataset_y]
    print(dataset_y)
    if len(dataset_y.shape) != 2: 
        dataset_y = torch.eye(torch.unique(dataset_y).shape[0])[dataset_y]
    print(dataset_y.shape)
    print(dataset_x.shape)
    dataset = data.TensorDataset(dataset_x, dataset_y)

    return data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

def dataLoadercloseSource(path, batchsize):
    dataset = np.load(path)
    dataset_x, dataset_y = torch.tensor(dataset['x'],dtype=torch.float32).reshape(-1, 1, 5000), torch.tensor(dataset['y'], dtype=torch.int)
    # dataset_y = torch.eye(K)[dataset_y]
    if dataset_y.shape == 2:
        dataset_y = torch.argwhere(dataset_y == 1)[:, 1].reshape(-1)
    print(dataset_y.shape)
    if len(dataset_y.shape) != 2: 
        dataset_y = torch.eye(torch.unique(dataset_y).shape[0] + 1)[dataset_y]
    print(dataset_y.shape)
    print(dataset_x.shape)
    dataset = data.TensorDataset(dataset_x, dataset_y)

    return data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

def dataLoadefeatures(path, batchsize):
    dataset = np.load(path)
    dataset_x, dataset_y = torch.tensor(dataset['feature'],dtype=torch.float32).reshape(-1, 256), torch.tensor(dataset['label'], dtype=torch.float32)
    # dataset_y = torch.eye(K)[dataset_y]
    print(dataset_y)
    if len(dataset_y.shape) != 2: 
        dataset_y = torch.eye(torch.unique(dataset_y).shape[0])[dataset_y]
    print(dataset_y.shape)
    print(dataset_x.shape)
    dataset = data.TensorDataset(dataset_x, dataset_y)

    return data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
