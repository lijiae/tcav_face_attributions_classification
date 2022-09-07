from torch.utils.data import Dataset
import os
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def interpolate(img, size):
    if type(size) == tuple:
        assert size[0] == size[1]
        size = size[0]

    orig_size = img.size(3)
    if size < orig_size:
        mode = 'area'
    else:
        mode = 'bilinear'
    return F.interpolate(img, (size, size), mode=mode)


def read_img(oimg):
    # img = Image.open(path).convert('RGB')
    img = TF.to_tensor(oimg)
    # img = img.unsqueeze(0)
    if img.size(-1) != 1024:
        img = interpolate(img, 1024)
    return img

class TrainDataset(Dataset):
    def __init__(self, class_list, origin_dataloader):
        self.pixel = []
        self.label = []
        for data, label in origin_dataloader:
            bs = data.size(0)
            for i in range(bs):
                x = data[i]
                l = label[i]
                if l in class_list:
                    self.pixel.append(x)
                    self.label.append(l)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.pixel[idx], self.label[idx]

# input image，输入类型dataloader
class CelebADataset(Dataset):

    def __init__(self,dl):
        self.dl=dl

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, item):
        # Sample
        sample = self.dl[item]

        # data and label information
        data = sample[0]
        label = sample[1]

        img=read_img(data)

        return (img,label)


class ConceptDataset(Dataset):
    def __init__(self,path,name):
        self.basedir=path
        nameslist=os.listdir(path)
        images=[]
        for name in nameslist:
            img=Image.open(os.path.join(path,name)).convert('RGB')
            images.append(img)
        self.images=images
        self.name=name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return (read_img(self.images[item]),self.name)
