from torch.utils.data import Dataset, DataLoader

class ModmaDataset(Dataset):
    def __init__(self, path, flag='train'):
        self.path = path
        self.flag = flag


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]