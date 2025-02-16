import json
import os
import cv2
import torch
import csv

from net.Retinex.retinex import *
from net.Retinex.retinex2 import *
from net.LIME.exposure_enhancement import enhance_image_exposure
from net.AHE.ahe import AHE
import json
from utils.myplot import *
from utils.metrics import *
from utils.utils import list_directories, list_files


class Experiment(object):
    """
    图像增强实验
    """

    def __init__(self, **kwargs) -> None:
        self.data_path = os.path.abspath(kwargs['root'])
        self.pic_save_path = os.path.abspath(kwargs['pic_save_path'])
        self.single_img, self.single_img_path = self._get_single_image()
        self.is_save = kwargs['is_save_pic']
        self.metric_save_path = kwargs['metric_save_path']
        self.lime_params = kwargs['LIME']

    def _get_single_image(self):
        img_list = os.listdir(self.data_path)
        if len(img_list) == 0:
            print('Data directory is empty.')
            exit()

        for img_name in img_list:
            if img_name == '.gitkeep':
                continue

            img_path = os.path.abspath(os.path.join(self.data_path, img_name))
            image = cv2.imread(img_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, img_path

    def retinex(self) -> None:
        with open("config/retinex_config.json", 'r') as f:
            config = json.load(f)

        img_msrcr = MSRCR(
            self.single_img,
            config['sigma_list'],
            config['G'],
            config['b'],
            config['alpha'],
            config['beta'],
            config['low_clip'],
            config['high_clip']
        )

        img_amsrcr = automatedMSRCR(
            self.single_img,
            config['sigma_list']
        )

        img_msrcp = MSRCP(
            self.single_img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']
        )

        shape = self.single_img.shape
        cv2.imshow('Image', self.single_img)
        cv2.imshow('retinex', img_msrcr)
        cv2.imshow('Automated retinex', img_amsrcr)
        cv2.imshow('MSRCP', img_msrcp)
        key = cv2.waitKey(0)
        if key == ord('s'):
            save_path = os.path.join(self.pic_save_path, 'retinex')
            print(f"保存路径：{save_path}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            retinex_save_path = os.path.join(save_path, "MSRCR.jpg")
            auto_retinex_save_path = os.path.join(save_path, "Auto_retinex.jpg")
            MSRCP_save_path = os.path.join(save_path, "MSRCP.jpg")
            cv2.imwrite(retinex_save_path, img_msrcr)
            cv2.imwrite(auto_retinex_save_path, img_amsrcr)
            cv2.imwrite(MSRCP_save_path, img_msrcp)
        else:
            cv2.destroyAllWindows()

    def _get_image(self, images_list):
        img_list = []
        for image in images_list:
            if len(image.shape) == 3 and image.shape[2] == 3:  # 判断是否为彩色图像
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_list.append(image)
        return img_list

    def _save_image(self, images_list, image_name_list, save_path):
        for i, image in enumerate(images_list):
            save_name = os.path.join(save_path, image_name_list[i] + '.jpg')
            # print(f"保存路径{save_name}")
            cv2.imwrite(save_name, image)

    def _mk_dir_path(self, method:str):
        save_path = os.path.join(self.pic_save_path, method)
        print(f"保存路径：{save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def retinex_2(self, is_get=False):
        img_ssr_15 = retinex_SSR(self.single_img, 15)
        img_ssr_80 = retinex_SSR(self.single_img, 80)
        img_ssr_250 = retinex_SSR(self.single_img, 250)
        img_msr = retinex_MSR(self.single_img)
        img_msrcr = retinex_MSRCR(self.single_img)
        img_msrcp = retinex_MSRCP(self.single_img)
        img_amsr = retinex_AMSR(self.single_img)

        show_all_images([self.single_img, img_ssr_15, img_ssr_80, img_ssr_250, img_msr, img_msrcr, img_msrcp, img_amsr]
                        , ['Original', 'SSR(15)', 'SSR(80)', 'SSR(250)', 'MSR', 'MSRCR', 'MSRCP', 'AMSR'])

        if is_get:
            return self._get_image(
                [self.single_img, img_ssr_15, img_ssr_80, img_ssr_250, img_msr, img_msrcr, img_msrcp, img_amsr])

        if self.is_save:
            save_path = self._mk_dir_path('retinex_2')
            self._save_image([img_ssr_15, img_ssr_80, img_ssr_250, img_msr, img_msrcr, img_msrcp, img_amsr]
                             , ['SSR(15)', 'SSR(80)', 'SSR(250)', 'MSR', 'MSRCR', 'MSRCP', 'AMSR'], save_path)

    def eval_images_metrics(self, is_save_csv=True):
        if is_save_csv:
            csv_file = open(self.metric_save_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Image Name', 'PSNR', 'SSIM', 'LPIPS', 'NIQE'])
            csv_file.close()

        csv_file = open(self.metric_save_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)

        method_dir_list = list_directories(self.pic_save_path)
        for i, method_dir in enumerate(method_dir_list):
            images_list = list_files(method_dir)
            for path in images_list:
                img_path = os.path.join(method_dir, path)
                image_name = '_'.join(img_path.split("\\")[-2:]).split('.')[0]
                image = cv2.imread(img_path)
                m_psnr = psnr(self.single_img, image)
                m_ssim = ssim(self.single_img, image)
                m_lpips = lpips(self.single_img, image)
                m_niqe = cal_niqe(image)

                print(f"------------------------{image_name}-------------------------")
                print(f"PSNR:{m_psnr}")
                print(f"SSIM:{m_ssim}")
                print(f"lpips:{m_lpips}")
                print(f"niqe:{m_niqe}")

                csv_writer.writerow([image_name, m_psnr, m_ssim, m_lpips, m_niqe])
        csv_file.close()

    def lime(self):
        lime_img = enhance_image_exposure(self.single_img, self.lime_params['gamma'], self.lime_params['lambda_'], True,
                                                sigma=1, bc=1, bs=1, be=1, eps=1e-3)
        dual_img = enhance_image_exposure(self.single_img, self.lime_params['gamma'], self.lime_params['lambda_'], False,
                                                 sigma=1, bc=1, bs=1, be=1, eps=1e-3)

        if self.is_save:
            save_path = self._mk_dir_path('lime')
            self._save_image([lime_img, dual_img], ['LIME', 'DUAL'], save_path)

    def he_images(self):
        he = AHE(self.single_img)
        ghe_image = he.get_ghe()
        clahe_image = he.get_clahe()
        # lhe_image = he.layered_histogram_equalization()
        # bhe_image = he.bi_histogram_equalization()
        # celhe_image = he.local_histogram_equalization()

        if self.is_save:
            save_path = self._mk_dir_path('ahe')
            self._save_image([ghe_image, clahe_image],
                             ['ghe', 'clahe'],
                             save_path)



