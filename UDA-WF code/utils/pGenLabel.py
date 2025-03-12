import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
import math

    
def obtain_label(loader, netF, netB, device):
    start_test = True
    # 获取全部的数据，包括所有的特征，标签
    with torch.no_grad():
        for X in loader:
            # data = iter_test.next()
            inputs = X.to(device)
            feas = netF(inputs)
            outputs = netB(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                # all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                # all_label = torch.cat((all_label, labels.float()), 0)

    # 生成预测
    all_output = nn.Softmax(dim=-1)(all_output)
    _, predict = torch.max(all_output, 1)
    # print(np.unique(predict))
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    # 使用Cosine距离时，对特征进行处理
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1) 
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    # K = 100
    aff = all_output.float().cpu().numpy()
    # 初始化均值和方差
    mu = 0
    var = 1
    # index = np.argmin(aff, axis=0)
    # # print(index)
    # initc =  all_fea[index]

    # for _ in range(100):
    #     dd = cdist(all_fea, initc, 'cosine')
    #     pred_label = dd.argmin(axis=1)
    #     index = dd.argmin(axis=0).reshape(-1)
    #     initc = all_fea[index]
    #     min_dist = dd.min(axis=1)
    #     # print(min_dist.max())
    #     mu_b = min_dist.sum()/min_dist.shape[0] # 平均距离
    #     var_b = np.power((min_dist - mu_b),2).sum()/min_dist.shape[0] # 距离方差
    #     mu = 0.5 * mu + 0.5 * mu_b
    #     var = 0.5 * var + 0.5 * var_b




    for _ in range(20):
        initc = aff.transpose().dot(all_fea) # 其为聚类中心点，维度为 K C H
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

        # 只找对应的类
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>0)
        
        labelset = labelset[0]
        # print(labelset.shape)
        

        dd = cdist(all_fea, initc[labelset], 'cosine')
        # print(np.where(dd == np.nan))
        pred_label = dd.argmin(axis=-1)
        # np.where(pred_label)
        # index, counts= np.unique(pred_label,return_counts=True)
        # counts = np.array(counts, dtype=np.float32)
        # var = np.std(counts)
        # # # print(var,counts.mean())
        # # counts -= counts.mean()
        
        # # counts /= var
        # # print(counts)
        # mu_counts = counts.mean()
        # var_counts = np.power(counts.std(), 2)
        # # # print(mu_counts, var_counts)
    
        # # # print(mu_counts,var_counts)
        # # # np.sqrt(1/(2*np.pi * var_counts)) *
        # counts = np.exp(-np.power((counts-mu_counts),2)/(2 * var_counts))
        # print(counts)
        # print(counts)
        # print(counts.min(), counts.max(), counts)
        # counts = counts/counts.sum()
        # print(counts)
        
        predict = labelset[pred_label]
        
        # EMA 指数平均移动
        min_dist = dd.min(axis=-1)
        # print(min_dist.max())
        mu_b = min_dist.sum()/min_dist.shape[0] # 平均距离
        var_b = np.power((min_dist - mu_b),2).sum()/min_dist.shape[0] # 距离方差
        mu = 0.5 * mu + 0.5 * mu_b
        var = 0.5 * var + 0.5 * var_b
        
        # print(counts)
        aff = np.eye(K)[predict]
        # aff = np.eye(K)
        # aff[index] = aff[index] * counts.reshape(-1, 1)
        # aff = aff[predict]
   
    # UA_dist = 0.01 * dd/dd.mean(axis=0)
    # UA_dist = UA_dist/UA_dist.sum(keepdims=True)
    # min_dist = UA_dist.min(axis=1)
    #计算高斯分布作为权值 np.sqrt(1/(2*np.pi * var))
    GussianDis = np.exp(-np.power((min_dist-mu),2)/(2*var))
    # Label_weight = np.ones_like(min_dist)
    # print(min_dist)
    # Label_weight = GussianDis
    Label_weight = np.where(min_dist <= mu, 1, GussianDis)
    print(Label_weight.min(), Label_weight.max(), mu, var)
    # Label_weight = np.ones(predict.shape[0])

    # acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    label = np.eye(K)[predict]
    
    return label, Label_weight

    # 得到输出标签空间的熵值
    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    # ent = ent.float().cpu()

    # from sklearn.cluster import KMeans
    # kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    # labels = kmeans.predict(ent.reshape(-1,1))

    # 根据熵值进行聚类，只对熵值较小的数据生成伪标签
    # idx = np.where(labels==1)[0]
    # iidx = 0
    # if ent[idx].mean() > ent.mean():
    #     iidx = 1
    # known_idx = np.where(kmeans.labels_ != iidx)[0]

    # all_fea = all_fea[known_idx,:]
    # all_output = all_output[known_idx,:]
    # predict = predict[known_idx]
    # all_label_idx = all_label[known_idx]
    # ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    # all_fea = all_fea.float().cpu().numpy()
    # K = all_output.size(1)
    # aff = all_output.float().cpu().numpy()
    # initc = aff.transpose().dot(all_fea)
    # initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    # cls_count = np.eye(K)[predict].sum(axis=0)
    # labelset = np.where(cls_count>args.threshold)
    # labelset = labelset[0]

    # dd = cdist(all_fea, initc[labelset], args.distance)
    # pred_label = dd.argmin(axis=1)
    # pred_label = labelset[pred_label]

    # for round in range(1):
    #     aff = np.eye(K)[pred_label]
    #     initc = aff.transpose().dot(all_fea)
    #     initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    #     dd = cdist(all_fea, initc[labelset], args.distance)
    #     pred_label = dd.argmin(axis=1)
    #     pred_label = labelset[pred_label]

    # guess_label = args.class_num * np.ones(len(all_label), )
    # guess_label[known_idx] = pred_label

    # acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    # log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)

    # return guess_label.astype('int'), ENT_THRESHOLD