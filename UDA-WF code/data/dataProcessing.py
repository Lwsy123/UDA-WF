import numpy as np

dataset = np.load("./Dataset/AWF_100.npz")

dataset_x, dataset_y = dataset['x'], dataset['y']

labels = np.unique(dataset_y)
labels = np.eye(labels.shape[0])
AWF_x = [np.empty([0, 5000],dtype=int)]*4 # 得到空数据
AWF_y = [np.empty([0, labels.shape[0]],dtype=int)]*4


for i in range(100):

    index = np.argwhere(dataset_y == i).reshape(-1)

    for col, j in enumerate([25, 50, 75, 100]):
        index_select = np.random.choice(index, j, replace=False)
        x = dataset_x[index_select] # 得到当前的数据
        y = labels[np.ones(j,dtype=int) * i] # 得到当前的label
        AWF_x[col] = np.append(AWF_x[col], x, axis=0) # 追加数据 
        AWF_y[col] = np.append(AWF_y[col], y, axis=0) # 追加标签数据
        print(AWF_x[col].shape, AWF_y[col].shape)

for col, j in enumerate([25, 50, 75, 100]):
    np.savez("./Dataset/processed/AWF" + str(j), x = AWF_x[col], y = AWF_y[col])  

        
        
    
    


