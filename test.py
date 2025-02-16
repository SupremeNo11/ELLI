import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from niqe import niqe
import lpips
from sklearn.metrics import mean_squared_error
import skimage

def cal_ssim_psnr(path1, path2):  # 一次性读一个文件夹
    img_low = os.listdir(path1)
    img_high = os.listdir(path2)

    cal_psnr = []
    cal_ssim = []
    for i in range(15):
        # print(path1+'/'+ img_low[i])
        img1 = cv2.imread(path1 + '/' + img_low[i])
        img2 = cv2.imread(path2 + '/' + img_high[i])
        cal_psnr.append(psnr(img1, img2))
        cal_ssim.append(ssim(img1, img2, channel_axis=2))
    return np.array(cal_psnr).mean(), np.array(cal_ssim).mean()


def cal_niqe(path1, path2):  # 一次性读一个文件夹
    img_low = os.listdir(path1)
    img_high = os.listdir(path2)

    cal_niqe = []
    for i in range(15):
        ref = np.array(Image.open(path1 + '/' + img_low[i]).convert('LA'))[:, :, 0]  # ref
        cal_niqe.append(niqe(ref))
    return np.array(cal_niqe).mean()


def cal_lpips(path1, path2):  # 一次性读一个文件夹
    img_low = os.listdir(path1)
    img_high = os.listdir(path2)

    use_gpu = True  # Whether to use GPU
    spatial = True
    cal_lpips = []
    for i in range(15):
        loss_fn = lpips.LPIPS(net='alex', spatial=spatial)
        if (use_gpu):
            loss_fn.cuda()
        dummy_img0 = lpips.im2tensor(lpips.load_image(path1 + '/' + img_low[i]))
        dummy_img1 = lpips.im2tensor(lpips.load_image(path2 + '/' + img_low[i]))
        dummy_img0 = dummy_img0.cuda()
        dummy_img1 = dummy_img1.cuda()
        dist = loss_fn.forward(dummy_img0, dummy_img1)
        cal_lpips.append(dist.mean().item())
    return np.array(cal_lpips).mean()


def cal_rmse(path1, path2):  # 一次性读一个文件夹
    img_low = os.listdir(path1)
    img_high = os.listdir(path2)
    cal_rmse = []
    for i in range(15):
        rms = mean_squared_error(path1 + '/' + img_low[i], path1 + '/' + img_high[i], squared=False)
        cal_rmse.append(rms)
    return np.array(cal_rmse).mean()


def cal_deltae(path1, path2):  # 一次性读一个文件夹
    img_low = os.listdir(path1)
    img_high = os.listdir(path2)
    cal_deltae = []
    for i in range(15):
        deltae = skimage.color.deltaE_cie76(path1 + '/' + img_low[i], path1 + '/' + img_high[i])
        cal_deltae.append(deltae)
    return np.array(cal_deltae).mean()


def pi(path1, path2):
    img_low = os.listdir(path1)
    img_high = os.listdir(path2)
    cal_pi = []
    for i in range(15):
        cal_pi.append(pi)
    return np.array(cal_pi).mean()


def cal_all(path_high, path, txt_path):
    print('computing...')
    file = open(txt_path, 'w', encoding='utf-8')
    file_list = os.listdir(path)
    c_psnr = []
    c_ssim = []
    c_niqe = []
    c_lpips = []
    c_rmse = []
    c_deltae = []
    for i in file_list:
        print(i)
        a, b = cal_ssim_psnr(path + i, path_high)
        # c=cal_fsim(path+i,path_high)
        c = 0
        d = cal_niqe(path + i, path_high)
        e = cal_lpips(path + i, path_high)
        f = cal_rmse(path + i, path_high)
        g = cal_deltae(path + i, path_high)
        c_psnr.append(a)
        c_ssim.append(b)
        c_niqe.append(d)
        c_lpips.append(e)
        c_rmse.append(f)
        c_deltae.append(g)
        print(i + ' psnr:' + str(a) + '    ' + '    ' + ' ssim:' + str(b) + '    ' + 'fsim: ' + str(
            c) + '    ' + ' niqe:' + str(d) + '    ' + 'lpips: ' + str(e) + '    ' + ' rmse:' + str(
            f) + '    ' + 'deltae: ' + str(g) + '\n')
    file.close()


print('sss')
cal_all(path_high=r"F:\DataSpace\ELLI\test\original\\",
        path=r"F:\DataSpace\ELLI\test\enhance\\",
        txt_path=r"F:\LLimg\ELLI\output\txt\out_data_outdoor01.txt")  # 文件放这
# 其中：gt是一级文件夹，里面直接是不同的图片
# 增强结果是二级文件夹，下一级是不同方法的文件夹，再下一级是不同的图片
# 为什么每次循环是15：因为lol的test长度是15

