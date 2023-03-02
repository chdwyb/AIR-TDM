import os
import cv2
from metrics import calculate_psnr, calculate_ssim
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# 自定义
from options import Options

opt = Options()
path_result = opt.Result_Path_Test
path_target = opt.Target_Path_Test
image_list = os.listdir(path_target)
L = len(image_list)
print(L)
psnr, ssim = 0, 0

for i in range(L):
    image_in = cv2.imread(path_result+str(image_list[i]), cv2.IMREAD_COLOR)
    image_tar = cv2.imread(path_target+str(image_list[i]), cv2.IMREAD_COLOR)
    ps = calculate_psnr(image_in, image_tar, test_y_channel=True)
    ss = calculate_ssim(image_in, image_tar, test_y_channel=True)
    psnr += ps
    ssim += ss
    print(i)
print(psnr/L, ssim/L)