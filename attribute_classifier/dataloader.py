from torch.utils.data import DataLoader,Dataset

class imgori(Dataset):
    def __init__(self,data):

        self.data=data

    def __getitem__(self, item):
        return self.data[item][0],self.data[item][1]


    def __len__(self):
        return len(self.data)