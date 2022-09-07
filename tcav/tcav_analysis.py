import numpy as np


def load_result(path):
    result=np.load(path)
    rlist=[]
    token=np.isnan(result[0])
    token_num=np.argwhere(~token)

    for i in range(6):
        attr=result[i]
        attr=attr[~token]
        rlist.append(attr)
    return np.array(rlist),token_num



result_path='/home/lijia/codes/anonymization/TCAV_PyTorch/tcav/output/tcav_result.npy'
result,token=load_result(result_path)
anay=np.sum(result,axis=1)
print(anay)