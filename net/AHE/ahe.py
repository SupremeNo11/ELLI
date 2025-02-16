import cv2
import numpy as np
from utils.tools import measure_time


class AHE:
    def __init__(self, image):
        self.image = image

    @measure_time
    def get_ghe(self):
        """全局直方图均衡化 (Global Histogram Equalization, GHE)"""
        b_img = cv2.equalizeHist(self.image[:, :, 0])
        g_img = cv2.equalizeHist(self.image[:, :, 1])
        r_img = cv2.equalizeHist(self.image[:, :, 2])

        return np.dstack((b_img, g_img, r_img))

    @measure_time
    def get_clahe(self):
        """对比度限制自适应直方图均衡化 (Contrast Limited Adaptive Histogram Equalization, CLAHE)"""

        m_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b_img = m_clahe.apply(self.image[:, :, 0])
        g_img = m_clahe.apply(self.image[:, :, 1])
        r_img = m_clahe.apply(self.image[:, :, 2])

        return np.dstack((b_img, g_img, r_img))

    # @measure_time
    # def layered_histogram_equalization(self, layers=3):
    #     """分层直方图均衡化 (Layered Histogram Equalization, LHE)"""
    #     img = self.image.astype(np.float32) / 255.0
    #     layer_images = []
    #     current_img = img.copy()
    #
    #     for i in range(layers):
    #         blurred_img = cv2.GaussianBlur(current_img, (5, 5), 2 ** i)
    #         layer = current_img - blurred_img
    #         current_img = blurred_img
    #         layer_images.append(layer)
    #
    #     layer_images.append(current_img)
    #     enhanced_layers = [cv2.equalizeHist((layer * 255).astype(np.uint8)) for layer in layer_images]
    #
    #     enhanced_img = sum(enhanced_layers) / layers
    #     enhanced_img = np.clip(enhanced_img, 0, 1)
    #     return (enhanced_img * 255).astype(np.uint8)
    #
    # @measure_time
    # def bi_histogram_equalization(self):
    #     """双直方图均衡化 (Bi-Histogram Equalization, BHE)"""
    #     median_intensity = np.median(self.image)
    #     low_intensity = self.image[self.image <= median_intensity]
    #     high_intensity = self.image[self.image > median_intensity]
    #
    #     low_equalized = cv2.equalizeHist(low_intensity)
    #     high_equalized = cv2.equalizeHist(high_intensity)
    #
    #     result = self.image.copy()
    #     result[self.image <= median_intensity] = low_equalized
    #     result[self.image > median_intensity] = high_equalized
    #
    #     return result
    #
    # @measure_time
    # def local_histogram_equalization(self, block_size=8):
    #     """对比度局部均衡化 (Contrast Enhancement using Local Histogram Equalization)"""
    #     img_height, img_width = self.image.shape
    #     result = np.zeros_like(self.image)
    #
    #     for i in range(0, img_height, block_size):
    #         for j in range(0, img_width, block_size):
    #             block = self.image[i:i + block_size, j:j + block_size]
    #             equalized_block = cv2.equalizeHist(block)
    #             result[i:i + block_size, j:j + block_size] = equalized_block
    #
    #     return result
