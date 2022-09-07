"""
陈若愚
2022.06.22

测试属性分类器的性能。
"""

import cv2
import torch
import argparse
import os
import numpy as np

import sys

sys.path.append("../")

from attribute_classifier import BranchedTiny,BranchedTinyAttr
from tools import load_yaml, read_img
from tqdm import tqdm
import random
from attribute_classifier.dataloader import imgori
from torch.utils.data import DataLoader

def parse_args():
    """
    程序分为两个步骤，转换与测试：

    第一个步骤，首先从json-path中获取我们需要测试的图像名称，并且从
    属性数据集attribute-set中查找是否由这样的标签，如果有，将数据
    写入到文件test-set中。
    转换步骤可以选择是否转换，如果先前已经转换完成了，可将参数
    convert-data设置为False，这样如果已经存在test-set文件，程序
    将不会进一步convert数据。当然如果想强制转换，可以设置convert-data
    为True

    第二个步骤，测试属性准确率。
    在给定test-set文件后（可以指定，也可以通过步骤一生成），可以通过
    第二部程序生成准确率。首先需要指定测试图像的文件夹image-dir，例如
    原始的VGGFace2还是VGGFace2Hq需要指定。然后需要指定测试的属性attribute
    之后就可以进行属性的测试了，给出一个准确率。目前只支持单个属性。

    如果测试集太大可以设置test-number来指定有效的测试数据，如果全部测试这个
    参数默认置1就行。

    指定运算设备，device。可以指定"cpu"，或者想要的gpu: "cuda:3"。目前版本不支持多卡运算
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=str,
                        default='Male',
                        choices=[  # 属性名字请从这里选，选1个
                            "5_o_Clock_Shadow", 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                            'Bald Bangs Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                            'Heavy_Makeup',
                            'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                            'Oval_Face Pale_Skin',
                            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                            'Wearing_Necktie', 'Young'
                        ])
    parser.add_argument('--image-dir', type=str,
                        default='/home/lijia/datasets/face/CelebAMask-HQ/CelebA-HQ-img')  # VGGFace2数据集的文件夹，这个放置固定的
    parser.add_argument('--attribute-net', type=str, default="BranchedTinyAttr",
                        choices=["VGGAttributeNet", "BranchedTinyAttr"])  # 属性网络
    parser.add_argument('--attribute-set', type=str, default='/exdata2/RuoyuChen/Datasets/VGGFace2/attribute')  # 属性标签
    parser.add_argument('--test-set', type=str,
                        default='/home/lijia/codes/202208/tcav_face/attribute_classifier/test_id.txt')  # 属性测试集的list
    parser.add_argument('--json-path', type=str,
                        default='/exdata2/RuoyuChen/Demo/MaskFaceGAN/json_per_ima')  # json path
    parser.add_argument('--convert-data', type=bool, default=False)  # 是否需要转换数据，如果数据已经转换并且系统检索到了，就不需要花这个时间继续转了
    parser.add_argument('--test-number', type=int, default=-1)  # 整个测试集非常大，可以选择其中的部分，测试全部的话置-1

    parser.add_argument('--device', type=str, default='cpu')  # json path

    args = parser.parse_args()
    return args


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

def Path_Image_Preprocessing(path):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    image = cv2.imread(path)
    if image is None:
        return None
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)
    image -= mean_bgr
    # H * W * C   -->   C * H * W
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image



def main(args):
    """
    计算某个属性的预测准确率。
    """
    device = args.device  # "cpu"

    #if args.attribute_net == "BranchedTinyAttr"
    cfg = load_yaml('/home/lijia/codes/202208/tcav_face/attribute_classifier/config.yaml')
    # model = BranchedTinyAttr(cfg.MODELS.CLASSIFIER)
    # model.set_idx_list(attributes=[VGG2CelebA[args.attribute]])

    model=BranchedTiny(ckpt=cfg.MODELS.CLASSIFIER.CKPT)
    model.eval()
    model.to(device)

    # 开始读测试集
    f = open(args.test_set)
    datas = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
    f.close()  # 关
    # random.shuffle(datas)

    attr_idx = get_idx(args.attribute)

    test_data=[]
    # 循环数据
    for data in tqdm(datas):
        data = data.strip()  # 正则，去除空格
        im_name = data.split()[0]
        image_path = os.path.join(
            args.image_dir, im_name
        )
        # print(image_path)
        if not os.path.exists(image_path):
            continue
        attr_label = int(data.split()[attr_idx + 1])

        input_data = read_img(image_path)
        if input_data.size(-1) != 224:
            input_data = F.interpolate({'size': (224,224), 'mode': 'area'})
        test_data.append([input_data.squeeze(0),attr_label])

    dataset=imgori(test_data)
    dl=DataLoader(dataset,batch_size=32)

    sum_score=0
    count=0
    # input_data =dl.to(device)
    for l in dl:
        count=count+1
        input_image=l[0]
        labels=l[1]
        input_image=input_image.to(device)
        predicted = model(input_image)
        threshold=0.8
        predicted = torch.sigmoid(predicted.unsqueeze(-1)[:,get_branched_id(args.attribute),:]).squeeze(1)
        # predicted = torch.sigmoid(predicted).squeeze(1)
        predicted_label=predicted
        predicted_label[predicted>threshold]=1
        predicted_label[predicted<=threshold]=-1


        result=predicted_label==labels
        result[result==True]=1
        result[result==False]=0

        sum_score=sum_score+result.numpy().sum()/len(result)
        print('{}batch socre:{}'.format(count,result.numpy().sum()/len(result)))

    print(sum_score/count)
        # predicted_truth = predicted[0][0].item() > 0.8

    # print("测试结束，有效测试数据数量为{}张，属性{}的预测结果的TP为{}，ACC为{}".format(number, args.attribute, TP / TP_num,
    #                                                                                 acc / number))


if __name__ == '__main__':

    args = parse_args()  # n, gpu


    # 测试准确率
    main(args)