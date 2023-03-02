import sys
import time
import math
from tqdm import tqdm  # 进度条
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
from skimage import img_as_ubyte
from torchvision.utils import save_image
import torch.nn.functional as F
# 自定义
from model import Uformer
from datasets import *
from options import Options


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask


if __name__ == '__main__':

    opt = Options()  # 超参数配置

    inputPathTest = opt.Input_Path_Test  # 测试输入图片路径
    # targetPathTest = opt.Target_Path_Test  # 测试目标图片路径
    resultPathTest = opt.Result_Path_Test  # 测试目标图片路径

    myNet = Uformer(img_size=128,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True, dd_in=3)  # 实例化网络
    myNet = nn.DataParallel(myNet)
    if opt.CUDA_USE:
        myNet = myNet.cuda()  # 网络放入GPU中

    # 测试数据
    datasetTest = MyTestDataSet(inputPathTest)  # 实例化测试数据集类
    # 可迭代数据加载器加载测试数据
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True)

    # 测试
    print('--------------------------------------------------------------')
    # 加载已经训练好的模型参数
    # 带下划线的是 Restormer
    if opt.CUDA_USE:
        myNet.load_state_dict(torch.load('./model.pth'))
    else:
        myNet.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
    myNet.eval()  # 指定网络模型测试状态

    with torch.no_grad():  # 测试阶段不需要梯度
        timeStart = time.time()  # 测试开始时间
        for index, (x, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()  # 释放显存

            input_test = x.cuda() if opt.CUDA_USE else x   # 放入GPU
            _, _, h, w = input_test.shape
            input_test, mask = expand2square(input_test, factor=128)
            restored = myNet(input_test)  # 输入网络，得到输出
            # print(restored, mask)
            restored = torch.masked_select(restored, mask.bool()).reshape(1, 3, h, w)
            restored = torch.clamp(restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

            cv2.imwrite(resultPathTest + name[0], cv2.cvtColor(img_as_ubyte(restored), cv2.COLOR_RGB2BGR))  # 保存网络输出结果
        timeEnd = time.time()  # 测试结束时间
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))
