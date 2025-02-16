import numpy as np
import cv2


def get_gauss_kernel(sigma, dim=2):
    """1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after
       normalizing the 1D kernel, we can get 2D kernel version by
       matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that
       if you want to blur one image with a 2-D gaussian filter, you should separate
       it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column
       filter, one row filter): 1) blur image with first column filter, 2) blur the
       result image of 1) with the second row filter. Analyse the time complexity: if
       m&n is the shape of image, p&q is the size of 2-D filter, bluring image with
       2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)"""
    ksize = int(np.floor(sigma * 6) / 2) * 2 + 1  # kernel size("3-σ"法则) refer to
    # https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
    k_1D = np.arange(ksize) - ksize // 2   # 0-90 / 45
    k_1D = np.exp(-k_1D ** 2 / (2 * sigma ** 2))
    k_1D = k_1D / np.sum(k_1D)
    if dim == 1:
        return k_1D
    elif dim == 2:
        return k_1D[:, None].dot(k_1D.reshape(1, -1))


def gauss_blur_original(img, sigma):
    """suitable for 1 or 3 channel image"""
    row_filter = get_gauss_kernel(sigma, 1)
    t = cv2.filter2D(img, -1, row_filter[..., None])
    return cv2.filter2D(t, -1, row_filter.reshape(1, -1))


def gauss_blur_recursive(img, sigma):
    """refer to “Recursive implementation of the Gaussian filter”
       (doi: 10.1016/0165-1684(95)00020-E). Paper considers it faster than
       FFT(Fast Fourier Transform) implementation of a Gaussian filter.
       Suitable for 1 or 3 channel image"""
    pass


def gauss_blur(img, sigma, method='original'):
    if method == 'original':
        return gauss_blur_original(img, sigma)
    elif method == 'recursive':
        return gauss_blur_recursive(img, sigma)