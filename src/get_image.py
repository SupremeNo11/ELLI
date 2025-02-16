import os
import cv2
from net.Retinex.retinex2 import *
from utils.myplot import *


def get_single_image(dir_path):
    img_list = os.listdir(dir_path)
    if len(img_list) == 0:
        print('Data directory is empty.')
        exit()

    for img_name in img_list:
        if img_name == '.gitkeep':
            continue

        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        return img


if __name__ == '__main__':
    img = get_single_image(r"../data")
    cv2.imshow('original_image', img)
    show_image(img, 'original_image')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
