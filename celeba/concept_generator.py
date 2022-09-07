import os
import argparse
import random
import cv2


celeba_attribution=["5_o_Clock_Shadow", 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                    'Bushy_Eyebrows', 'Chubby' ,'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                    'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
                    ]

def get_idx(attr_name):
    if attr_name in celeba_attribution:
        return celeba_attribution.index(attr_name)
    return -1

def is_positive(ids,attrs):
    for id in ids:
        if int(attrs[id+1])==1:
            continue
        else:
            return False
    return True

def is_negative(ids,attrs):
    for id in ids:
        if int(attrs[id+1])==-1:
            continue
        else:
            return False
    return True

def concept_in_file(args,savepath,imglist):
    with open(savepath,'w') as f:
        for img in imglist:
            f.writelines(os.path.join(args.root,img)+'\n')

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=str,default='/home/lijia/datasets/face/CelebAMask-HQ/CelebA-HQ-img',help='which is the root of celeba images')
    parser.add_argument('--anno_root',type=str,default='/home/lijia/datasets/face/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt',help='the attribution annotation txt')
    parser.add_argument('--positive_attr',type=str,default='Male',help='the attributions you want to set positive pair,'
                                                             'use space to split attributions')
    parser.add_argument('--negative_attr',type=str,default='Male',help='the attributions you want to set positive pair,'
                                                             'use space to split attributions')
    parser.add_argument('--count',type=int,default=100,help='the images in concepts')
    parser.add_argument('--binary_type',type=bool,default=True)
    parser.add_argument('--save_dir_txt',type=str,default='')

    args=parser.parse_args()
    return args

def get_attrs_idlist(attr):
    attrs = attr.split()
    attrs_id = []
    for a in attrs:
        a_id = get_idx(a)
        assert (a_id > 0), 'the input attribution is not in CelebA'
        attrs_id.append(a_id)
    return attrs_id

def main(args):
    # 得到anno的数据
    f = open(args.anno_root)
    annos = f.readlines()[2:]
    f.close()

    # 得到属性对应的id
    attrs_id=get_attrs_idlist(args.positive_attr)
    n_attrs_id=get_attrs_idlist(args.negative_attr)
    if len(n_attrs_id)<=0:
        n_attrs_id=attrs_id

    # 筛选
    po_list = []
    neg_list = []
    for n in annos:
        n = n.strip()
        ns = n.split()
        if is_positive(attrs_id, ns):
            po_list.append(ns[0])
        if is_negative(n_attrs_id, ns):
            neg_list.append(ns[0])

    cross_list=list(set(po_list).intersection(set(neg_list)))
    print('{} positive list:'.format(args.positive_attr), len(po_list))
    print('{} negative list:'.format(args.negative_attr), len(neg_list))

    if len(po_list) > args.count:
        random.shuffle(po_list)
        po_list = po_list[:args.count]
    if len(neg_list) > args.count:
        random.shuffle(neg_list)
        neg_list = neg_list[:args.count]

    concept_in_file(args, '{}.txt'.format(args.positive_attr.replace(' ', '_')), po_list)
    concept_in_file(args, '!{}.txt'.format(args.negative_attr.replace(' ', '_')), neg_list)
    if len(cross_list)!=0:
        if len(cross_list) > args.count:
            random.shuffle(cross_list)
            cross_list = cross_list[:args.count]
        concept_in_file(args, '{}_!{}.txt'.format(args.positive_attr.replace(' ', '_'),args.negative_attr.replace(' ', '_')), cross_list)


def generate_concept(txtpath,concept_path):
    if not os.path.exists(concept_path):
        os.makedirs(concept_path)

    with open(txtpath) as f:
        datas=f.readlines()
    i=0
    for d in datas:
        d=d.strip()
        img=cv2.imread(d)
        cv2.imwrite(os.path.join(concept_path,'{}.jpg'.format(str(i))),img)
        i=i+1


if __name__=='__main__':
    args=parse_args()

    main(args)

    generate_concept('/home/lijia/codes/202208/tcav_face/celeba/Male_Mustache.txt','/home/lijia/codes/202208/tcav_face/concepts/Male_Mustache')


