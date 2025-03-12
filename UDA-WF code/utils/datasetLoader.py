import torch 
from torch import nn
from torch.utils import data


class DatastLoader(data.Dataset):
    def __init__(self, data) -> None:
        super(DatastLoader, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        return torch.tensor(self.data[index], dtype=torch.float32)