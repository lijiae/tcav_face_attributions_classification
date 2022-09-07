import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tcav import TCAV
from model_wrapper import ModelWrapper
from celeba.dataset import CelebADataset,ConceptDataset
import os
import prettytable as pt
import numpy as np
from attribute_classifier.BranchedTiny import BranchedTinyAttr,BranchedTiny
import yaml
from easydict import EasyDict as edict
import argparse

VGG2CelebA = {
    "Sideburns": 'sideburns',
    "Bald": 'bald',
    "Goatee": 'goatee',
    "Mustache": 'mustache',
    "5 o Clock Shadow": '5_o_clock_shadow',
    "Arched Eyebrows": 'arched_eyebrows',
    "no beard": 'no_beard',
    "Male": 'male',
    "Black Hair": 'black_hair',
    "High Cheekbones": 'high_cheekbones',
    "Smiling": 'smiling',
    "Oval Face": 'oval_face',
    "Bushy Eyebrows": 'bushy_eyebrows',
    "Young": 'young',
    "Gray Hair": 'gray_hair',
    "Brown Hair": 'brown_hair',
    "Blond Hair": 'blond_hair',
    "Chubby": 'chubby',
    "Double Chin": 'double_chin',
    "Big Nose": 'big_nose',
    "Bags Under Eyes": 'bags_under_eyes',
    "Bangs": 'bangs',
    "Wavy Hair": 'wavy_hair',
    "Big Lips": 'big_lips',
    "Pointy Nose": 'pointy_nose',
    "Receding Hairline": 'receding_hairline',
}

branched_name = ['blurry', 'sideburns', 'wearing_earrings', 'bald', 'goatee', 'mustache',
                           '5_o_clock_shadow', 'arched_eyebrows', 'no_beard', 'heavy_makeup', 'male',
                           'wearing_lipstick', 'black_hair', 'high_cheekbones', 'smiling',
                           'mouth_slightly_open', 'oval_face', 'bushy_eyebrows', 'attractive',
                           'young', 'gray_hair', 'brown_hair', 'blond_hair', 'pale_skin', 'chubby',
                           'double_chin', 'big_nose', 'bags_under_eyes', 'wearing_necklace', 'wearing_necktie',
                           'rosy_cheeks', 'bangs', 'wavy_hair', 'straight_hair', 'wearing_hat', 'big_lips',
                           'narrow_eyes', 'pointy_nose', 'receding_hairline', 'eyeglasses']

celeba_attribution=["5_o_Clock_Shadow", 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                    'Bushy_Eyebrows', 'Chubby' ,'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                    'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
                    ]

def get_idx(attr_name):
    return celeba_attribution.index(attr_name)

def get_branched_id(attr_name):
    return branched_name.index(VGG2CelebA[attr_name])

def data_loader(base_path,dirname):
    image_dataset_train = ConceptDataset(base_path,dirname)
    train_loader = DataLoader(image_dataset_train, batch_size=1)
    return train_loader

def validate(model):
    #选取网络结构的层数，固定hook
    extract_layer = 'layers15'
    model = ModelWrapper(model, [extract_layer])

    #输入：模型，目标类data，conceptdata，输入类别信息
    scorer = TCAV(model, validloader, concept_dict, class_dict.values(),100)

    print('Generating concepts...')
    # 生成
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

def list_to_numpy(cfg):
    for k, v in cfg.items():
        if type(v) is list:
            cfg[k] = np.array(sorted(v))
    return cfg

def load_yaml(filename):
    with open(filename, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = list_to_numpy(cfg)
    return edict(cfg)

def parse_args():
    parse=argparse.ArgumentParser()
    parse.add_argument('--target_class',type=str,default='Male')
    parse.add_argument('--yaml_path',type=str,default='/home/lijia/codes/202208/tcav_face/attribute_classifier/config.yaml')
    parse.add_argument('--model_path',type=str,default='/home/lijia/codes/202208/tcav_face/attribute_classifier/BranchedTiny.ckpt')
    args=parse.parse_args()
    return args

if __name__ == "__main__":
    args=parse_args()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 输入图像
    image_dataset = datasets.ImageFolder('/home/lijia/codes/202208/tcav_face/images/classification/')

    class_dict = {
        args.target_class:get_branched_id(args.target_class)
    }

    reverse_class_dict = {v : k for k, v in class_dict.items()}

    validate_dataset = CelebADataset(image_dataset)
    validloader = DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=1)

    concept_dict = {}
    for dirname in os.listdir('/home/lijia/codes/202208/tcav_face/images/concepts'):
        fullpath = os.path.join('/home/lijia/codes/202208/tcav_face/images/concepts', dirname)
        if os.path.isdir(fullpath):
            concept_dict[dirname] = data_loader(fullpath,dirname)

    cfg = load_yaml(args.yaml_path)
    model = BranchedTiny(cfg.MODELS.CLASSIFIER.CKPT)
    model.eval()

    model.to(device)

    # train()
    validate(model)
