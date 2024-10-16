import copy
import random
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os
import torch
import numpy as np

class SeedDataset(Dataset):
    def custom_train_test_split(self,datas, labels, test_size=0.2, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        # 创建数据对，以便我们可以在不打乱标签顺序的情况下打乱数据
        data_label_pairs = list(zip(datas, labels))
        random.shuffle(data_label_pairs)

        # 计算测试集的大小
        test_set_size = int(len(datas) * test_size)

        # 分割数据对为训练集和测试集
        test_data, test_label = zip(*data_label_pairs[:test_set_size])
        train_data, train_label = zip(*data_label_pairs[test_set_size:])

        return list(train_data), list(test_data), list(train_label), list(test_label)
    
    def __init__(self, path='./seed', flag='train'):
        self.path = path
        self.flag = flag
        datas = []
        labels = []
        label_3s = [
            [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
        ]
        for n in range(3):
            for root, dirs, files in os.walk(path + '/'+str(n+1)):
                for file in files:
                    if file.endswith('.mat'):
                        mat_data = sio.loadmat(os.path.join(root, file))
                        keys = list(mat_data.keys())
                        for i in range(24):
                            d = np.array(mat_data[keys[i+3]])
                            datas.append(d[:, :8000])
                            labels.append(label_3s[n][i])
                    

        train_data, test_data, train_label, test_label = self.custom_train_test_split(datas, labels, test_size=0.2, random_state=42)

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
            data_one.append(data[index][0])  # fp1
            data_one.append(data[index][2])  # fp2
            data_one.append(data[index][25])  # c3
            data_one.append(data[index][29])  # c4
            data_one.append(data[index][58])  # o1
            data_one.append(data[index][60])  # o2
            data_one.append(data[index][41])  # p7
            data_one.append(data[index][49])  # p8

            data_one2d = []
            for i in range(8000):
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