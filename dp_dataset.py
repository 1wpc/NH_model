from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os

class ModmaDataset(Dataset):
    def __init__(self, path='./modma', flag='train'):
        self.path = path
        self.flag = flag
        mat_datas = []
        labels = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.mat'):
                    mat_datas.append(sio.loadmat(os.path.join(root, file)[file]))
                    if file.startswith('0201'):
                        labels.append(1)
                    else:
                        labels.append(0)



        self.data = []



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]