# -*- coding:utf-8 -*-
"""
# @Project   :ELLI
# @FileName  :PS.py
# @Time      :2025/1/12 10:40
# @Author    :Zhiyue Lyu
# @Version   :1.0
# @Descript  :该脚本主要对图像进行处理，去除空白的背景，抠出主题物体
"""

import cv2
import numpy as np
import os


def remove_background(image_path: str, output_path: str) -> None:
    """
    剔除背景色，尽量是白色背景
    :param image_path: 输入图片路径
    :param output_path: 结果保存路径
    :return: None
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取文件: {image_path}")
        return

    # 将图片从BGR转换到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设置更加精确的白色背景HSV范围
    lower_white = np.array([0, 0, 180])  # 适当调整S和V的范围，避免影响主体
    upper_white = np.array([180, 30, 255])

    # 创建一个掩码，检测白色区域
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 将图像添加alpha通道（透明度通道）
    image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # 转为带有Alpha通道的BGRA格式

    # 将掩码区域设为透明（alpha通道为0），其他区域保持不变
    image_with_alpha[mask == 0] = [0, 0, 0, 0]  # 背景部分设置为完全透明

    # 保存为PNG格式，保留透明背景
    cv2.imwrite(output_path, image_with_alpha)


if __name__ == "__main__":
    img_path = r"E:\CodeSpace\ELLI\script\shusongji.jpg"
    # 输出文件夹路径
    output_folder = 'outputs/'

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if img_path.endswith(".png") or img_path.endswith(".jpg"):
        output_path = os.path.join(output_folder, os.path.basename(img_path).replace('.jpg', '.png'))
        remove_background(img_path, output_path)

    print("图片处理完成！")