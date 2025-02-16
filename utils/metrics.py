import os

import cv2
import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.stats import norm
from skimage.util import view_as_windows
from niqe import niqe


# 初始化模型
loss_fn = lpips.LPIPS(net='alex')  # 可以选择'vgg'或'squeeze'，'alex'通常更快


def psnr(img1, img2, data_range=255):
    """使用第三方库计算PSNR指标"""
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def ssim(img1, img2, channel=2):
    """使用第三方库计算SSIM指标"""
    return structural_similarity(img1, img2, channel_axis=channel)


def lpips(img1, img2):
    # 将图像转换为Tensor并规范化
    img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 计算LPIPS
    lpips_value = loss_fn(img1_tensor, img2_tensor)

    return lpips_value.item()


def cal_niqe(image):  # 一次性读一个文件夹
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = niqe(gray_image)
    return np.array(score)

#
# # Function to calculate Ma score (dummy implementation)
# # Replace this with the actual Ma score calculation
# def calculate_ma_score(image):
#     """计算ma分数"""
#     # Placeholder for actual Ma score calculation
#     return 1 - (np.mean(image) / 255.0)  # This is just a dummy implementation
#
#
# # Function to calculate PI
# def PI(image):
#     """计算PI指标"""
#     niqe_score = niqe(image)
#     ma_score = calculate_ma_score(image)
#     pi_score = 0.5 * (niqe_score + (1 - ma_score) * 10)
#     return pi_score
