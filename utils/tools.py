import time
import numpy as np
from functools import wraps
import cv2
import os.path

eps = np.finfo(np.double).eps


def measure_time(wrapped):
    @wraps(wrapped)
    def wrapper(*args, **kwds):
        t1 = time.time()
        ret = wrapped(*args, **kwds)
        t2 = time.time()
        print('@measure_time: {0} took {1} seconds'.format(wrapped.__name__, t2 - t1))
        return ret

    return wrapper


def simplest_color_balance(img_msrcr, s1, s2):
    """see section 3.1 in “Simplest Color Balance”(doi: 10.5201/ipol.2011.llmps-scb).
    Only suitable for 1-channel image"""
    sort_img = np.sort(img_msrcr, None)
    N = img_msrcr.size
    Vmin = sort_img[int(N * s1)]
    Vmax = sort_img[int(N * (1 - s2)) - 1]
    img_msrcr[img_msrcr < Vmin] = Vmin
    img_msrcr[img_msrcr > Vmax] = Vmax
    return (img_msrcr - Vmin) * 255 / (Vmax - Vmin)