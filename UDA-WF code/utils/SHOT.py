import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss_SHOT
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def obtain_label_shot(loader, netF, netC, device):
    start_test = True
    with torch.no_grad():
       for X, y in loader:
            inputs = X.to(device)
            labels = y
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    # print(predict)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + 1e-6), dim=1) / np.log(100)
    ent = ent.float().cpu()

    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], "cosine")
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(10):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = 99 * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label
    guess_label = guess_label.astype('int')
    # print(guess_label)
    guess_label = np.eye(100, dtype=np.float32)[guess_label]

    # acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    # log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)

    return guess_label, ENT_THRESHOLD
