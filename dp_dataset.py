import copy
from sklearn.model_selection import train_test_split
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

        train_data, test_data, train_label, test_label = train_test_split(datas, labels, test_size=0.2, random_state=42)

        if self.flag == 'train':
            self.data = train_data
            self.label = train_label
        elif self.flag == 'test':
            self.data = test_data
            self.label = test_label
        else:
            raise ValueError("Invalid flag. Please use 'train' or 'test'.")
        
        self.data = self.choseEp(self.data)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).float()
        y = torch.tensor(self.label[idx])
        return x, y
    
    def choseEp(self,data:list):
        data_new = []
        for index in range(len(data)):
            data_one = []
            data_one.append(data[index][21])  # fp1
            data_one.append(data[index][8])  # fp2
            data_one.append(data[index][35])  # c3
            data_one.append(data[index][103])  # c4
            data_one.append(data[index][69])  # o1
            data_one.append(data[index][82])  # o2
            data_one.append(data[index][57])  # p7
            data_one.append(data[index][95])  # p8

            data_one2d = []
            for i in range(75000):
                p8 = []
                for j in range(8):
                    p8.append(data_one[j][i])
                image = []
                hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                hang[2] = p8[0]
                hang[3] = p8[1]
                image.append(copy.deepcopy(hang))
                hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                hang[1] = p8[2]
                hang[4] = p8[3]
                image.append(copy.deepcopy(hang))
                hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                hang[0] = p8[6]
                hang[5] = p8[7]
                image.append(copy.deepcopy(hang))
                hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                hang[2] = p8[4]
                hang[3] = p8[5]
                image.append(copy.deepcopy(hang))
                data_one2d.append(image)

            data_new.append(data_one2d)

        return data_new