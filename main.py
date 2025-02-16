import cv2
import os
import yaml
from src.Image_Enhance import Experiment
from utils.metrics import *
from niqe import niqe

with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file.read())

phase = "eval"


def main():
    exp = Experiment(**config)
    if phase == "retinex":
        exp.retinex()
    elif phase == "retinex2":
        images = exp.retinex_2()
    elif phase == "eval":
        exp.eval_images_metrics(is_save_csv=True)
    elif phase == "lime":
        exp.lime()
    elif phase == "ahe":
        exp.he_images()


if __name__ == '__main__':
    main()
