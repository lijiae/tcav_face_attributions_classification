import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets
from tcav import TCAV
from model_wrapper import ModelWrapper
import os
import prettytable as pt
import numpy as np
import pickle

from models.senet import senet50
from data.imagedata import ConceptDataset,ImageDataset


# 加载dict
def load_name_dict(path,count=500):
    namelist={}
    with open(path) as f:
        lines=f.readlines()
        lines=lines[:count]
        for line in lines:
            name_id=line.split(' ')
            namelist[name_id[0]]=int(name_id[1].split('\n')[0])
    print(namelist)
    return namelist

def data_loader(base_path):
    image_dataset_train = ConceptDataset(base_path)
    train_loader = DataLoader(image_dataset_train, batch_size=1)
    return train_loader


def validate(model):
    # model.load_state_dict(torch.load(checkpoints_path), strict=False)
    extract_layer = 'layer4'
    # 提取模型
    model = ModelWrapper(model, [extract_layer])

    # 计算分数，输入model，图像，
    scorer = TCAV(model, image_loader, concept_dict, class_dict.values(), 30000)

    print('Generating concepts...')
    scorer.generate_activations([extract_layer])
    scorer.load_activations()
    print('Concepts successfully generated and loaded!')

    print('Calculating TCAV scores...')
    scorer.generate_cavs(extract_layer)
    scorer.calculate_tcav_score(extract_layer, 'output/tcav_result.npy')
    scores = np.load('output/tcav_result.npy')
    scores = scores.T.tolist()
    print('Done!')

    table = pt.PrettyTable()
    table.field_names = ['class'] + list(concept_dict.keys())
    for i, k in enumerate(class_dict.keys()):
        new_row = [k] + scores[i]
        table.add_row(new_row)
    print(table)


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # class_dict=load_name_dict(path='/home/lijia/codes/anonymization/TCAV_PyTorch/data/name_id.txt')

    class_dict = {
        'n000002': 0,
        'n000003': 1,
        'n000004': 2,
        'n000005':3,

    }


    reverse_class_dict = {v : k for k, v in class_dict.items()}
    image_folder = datasets.ImageFolder('/home/lijia/codes/anonymization/TCAV_PyTorch/database/images')
    image_dataset=ImageDataset(image_folder)
    image_loader=DataLoader(image_dataset,batch_size=1, shuffle=False, num_workers=1)

    concept_dict = {}
    for dirname in os.listdir('../database/concepts_6'):
        fullpath = os.path.join('../database/concepts_6', dirname)
        if os.path.isdir(fullpath):
            concept_dict[dirname] = data_loader(fullpath)

    checkpoints_path='/home/lijia/codes/anonymization/TCAV_PyTorch/checkpoint/senet50_ft_weight.pkl'
    model = senet50(num_classes=8631)
    with open(checkpoints_path, 'rb') as f:
        weights={key:torch.from_numpy(arr) for key,arr in pickle.load(f, encoding='latin1').items()}
    model.load_state_dict(weights,strict=True)
    model.eval()
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    # train()
    validate(model)
