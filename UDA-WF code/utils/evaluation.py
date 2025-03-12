import numpy as np

def get_F1_Score(Precision, Recall):

    return 2 * (Precision * Recall) / (Precision + Recall) 

def get_Precision(TPlist, FPlist):
    
    return TPlist /(TPlist + FPlist)
    

def get_Recall(TPlist, FNlist):
    return TPlist / (TPlist + FNlist)
    

def evluation(y_pred,label):
    TPlist = []
    FPlist = []
    TNlist = []
    FNlist = []
    print(label.shape)
    if len(label.shape) == 2:
        label_alter = np.argwhere(label == 1)[:, 1].reshape(-1)
    else :
        label_alter = label.reshape(-1)
    if len(y_pred.shape) == 2:
        y_alter = np.argmax(y_pred, axis= -1).reshape(-1)
    else :
        y_alter = y_pred
    # print(y_alter.shape, label_alter.shape)
    for i in range(0, np.unique(label_alter).shape[0]):
        labelindex = np.argwhere(label_alter == i)
        yindex = np.argwhere(y_alter == i)

        TP = (y_alter[labelindex] == i).astype(np.float32).sum()
        FN = (y_alter[labelindex] != i).astype(np.float32).sum()
        FP = (label_alter[yindex] != i).astype(np.float32).sum()

        TPlist.append(TP)
        FNlist.append(FN)
        FPlist.append(FP)
    TPlist = np.nan_to_num(np.array(TPlist))
    FPlist = np.nan_to_num(np.array(FPlist))
    FNlist = np.nan_to_num(np.array(FNlist))
    # print(TPlist, FPlist, FNlist)
    Precision, Recall = np.nan_to_num(get_Precision(TPlist, FPlist)), np.nan_to_num( get_Recall(TPlist, FNlist))
    F1 = np.nan_to_num(get_F1_Score(Precision, Recall))
    return Precision.mean(), Recall.mean(), F1.mean()

