from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os
import torch

class ModmaDataset(Dataset):
    def __init__(self, path='./modma', flag='train'):
        self.path = path
        self.flag = flag
        datas = []
        labels = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.mat'):
                    mat_data = sio.loadmat(os.path.join(root, file))
                    datas.append(mat_data[list(mat_data.keys())[3]])#这个是ndarray类型的数据
                    if file.startswith('0201'):
                        labels.append(1)
                    else:
                        labels.append(0)

        self.data = datas
        self.label = labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.label[idx])
        return x, y