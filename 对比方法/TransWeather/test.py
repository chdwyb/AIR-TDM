import sys
import time
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
# 自定义
from model import Transweather
from datasets import *
from options import Options


# 打补丁
def pad(x, factor=16, mode='reflect'):
    _, _, h_even, w_even = x.shape
    padh_left = (factor - h_even % factor) // 2
    padw_top = (factor - w_even % factor) // 2
    padh_right = padh_left if h_even % 2 == 0 else padh_left + 1  # 如果原图分辨率是奇数，则打补丁右边和下边多一个像素
    padw_bottom = padw_top if w_even % 2 == 0 else padw_top + 1
    x = F.pad(x, pad=[padw_top, padw_bottom, padh_left, padh_right], mode=mode)
    return x, (padh_left, padh_right, padw_top, padw_bottom)


# 打补丁逆向
def unpad(x, pad_size):
    padh_left, padh_right, padw_top, padw_bottom = pad_size
    _, _, newh, neww = x.shape
    h_start = padh_left
    h_end = newh - padh_right
    w_start = padw_top
    w_end = neww - padw_bottom
    x = x[:, :, h_start:h_end, w_start:w_end]  # 切片
    return x


if __name__ == '__main__':

    opt = Options()  # 超参数配置

    inputPathTest = opt.Input_Path_Test  # 测试输入图片路径
    # targetPathTest = opt.Target_Path_Test  # 测试目标图片路径
    resultPathTest = opt.Result_Path_Test  # 测试目标图片路径

    myNet = Transweather()
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
        myNet.load_state_dict(torch.load('./model_best.pth'))
    else:
        myNet.load_state_dict(torch.load('./model_best.pth', map_location=torch.device('cpu')))
    myNet.eval()  # 指定网络模型测试状态

    with torch.no_grad():  # 测试阶段不需要梯度
        timeStart = time.time()  # 测试开始时间
        for index, (x, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()  # 释放显存

            input_test = x.cuda() if opt.CUDA_USE else x   # 放入GPU

            input_test, pad_size = pad(input_test, factor=16)  # 将输入补成 16 的倍数
            output_test = myNet(input_test)  # 输入网络，得到输出
            output_test = unpad(output_test, pad_size)  # 将补上的像素去掉，保持输出输出大小一致

            save_image(output_test, resultPathTest + name[0])  # 保存网络输出结果
        timeEnd = time.time()  # 测试结束时间
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))
