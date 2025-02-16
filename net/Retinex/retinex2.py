from utils.tools import measure_time, simplest_color_balance
from utils.gauss import gauss_blur
import numpy as np
from src.get_image import get_single_image
import cv2

eps = np.finfo(np.double).eps


@measure_time
def retinex_SSR(img, sigma):
    """log(R(x,y))=log(S(x,y))-log(S(x,y)*G(x,y))=log(S(x,y))-log(L(x,y)), i.e.
       r=s-l. S(x,y) and R(x,y) represent input image and retinex output image
       respectively, L(x,y):=S(x,y)*G(x,y) represents the lightness function,
       defined as the original image S operated with a gaussian filter G(named
       as center/surround function)
    implement ssr on single channel:
       1) read original image and convert to double(type) as S
       2) calc coefficient of G with sigma, i.e. normalize the gaussian kernel
       3) calc r by r=s-l and then convert r to R(from log to real)
       4) stretch the values of R into the range 0~255
    issue:
       we don't convert values from log domain to real domain in step 3 above,
       because it will bring terrible effect. In fact nobody does this, but the
       reason still remains unknown
    note:
       gauss blur is the main operation of SSR, its time complexity is O(mnpq),
       m&n is the shape of image, p&q is the size of filter, we can use recursive
       gaussian filter(RGF), O(mn), to alternative it(see func fast_gauss_blur).
       Or transform from time domain to frequency domain using Fourier Transform
       to reduce complexity[4]
    """
    if len(img.shape) == 2:
        img = img[..., None]
    ret = np.zeros(img.shape, dtype='uint8')
    for i in range(img.shape[-1]):
        channel = img[..., i].astype('double')
        S_log = np.log(channel + 1)
        gaussian = gauss_blur(channel, sigma)
        # gaussian=cv2.filter2D(channel,-1,get_gauss_kernel(sigma)) #conv may be slow if size too big
        # gaussian=cv2.GaussianBlur(channel,(0,0),sigma) #always slower
        L_log = np.log(gaussian + 1)
        r = S_log - L_log
        R = r  # R=np.exp(r)?
        mmin = np.min(R)
        mmax = np.max(R)
        stretch = (R - mmin) / (mmax - mmin) * 255  # linear stretch
        ret[..., i] = stretch
    return ret.squeeze()


@measure_time
def retinex_MSR(img, sigmas=[15, 80, 250], weights=None):
    """r=∑(log(S)-log(S*G))w, MSR combines various SSR with different(or same) weights,
       commonly we select 3 scales(sigma) and equal weights, (15,80,250) is a good
       choice. If len(sigmas)=1, equal to SSR
    args:
       sigmas: a list
       weights: None or a list, it represents the weight for each SSR, their sum should
          be 1, if None, the weights will be [1/t, 1/t, ..., 1/t], t=len(sigmas)
    """
    if weights is None:
        weights = np.ones(len(sigmas)) / len(sigmas)
    elif not abs(sum(weights) - 1) < 0.00001:
        raise ValueError('sum of weights must be 1!')
    ret = np.zeros(img.shape, dtype='uint8')
    if len(img.shape) == 2:
        img = img[..., None]
    for i in range(img.shape[-1]):
        channel = img[..., i].astype('double')
        r = np.zeros_like(channel)
        for k, sigma in enumerate(sigmas):
            r += (np.log(channel + 1) - np.log(gauss_blur(channel, sigma, ) + 1)) * weights[k]
        mmin = np.min(r)
        mmax = np.max(r)
        stretch = (r - mmin) / (mmax - mmin) * 255
        ret[..., i] = stretch
    return ret.squeeze()


def MultiScaleRetinex(img, sigmas=[15, 80, 250], weights=None, flag=True):
    '''equal to func retinex_MSR, just remove the outer for-loop. Practice has proven
       that when MSR used in MSRCR or Gimp, we should add stretch step, otherwise the
       result color may be dim. But it's up to you, if you select to neglect stretch,
       set flag as False, have fun'''
    if weights == None:
        weights = np.ones(len(sigmas)) / len(sigmas)
    elif not abs(sum(weights) - 1) < 0.00001:
        raise ValueError('sum of weights must be 1!')
    r = np.zeros(img.shape, dtype='double')
    img = img.astype('double')
    for i, sigma in enumerate(sigmas):
        r += (np.log(img + 1) - np.log(gauss_blur(img, sigma) + 1)) * weights[i]
    if flag:
        mmin = np.min(r, axis=(0, 1), keepdims=True)
        mmax = np.max(r, axis=(0, 1), keepdims=True)
        r = (r - mmin) / (mmax - mmin) * 255  # maybe indispensable when used in MSRCR or Gimp, make pic vibrant
        r = r.astype('uint8')
    return r


@measure_time
def retinex_MSRCR(img, sigmas=[12, 80, 250], s1=0.01, s2=0.01):
    """r=βlog(αI')MSR, I'=I/∑I, I is one channel of image, ∑I is the sum of all channels,
       C:=βlog(αI') is named as color recovery factor. Last we improve previously used
       linear stretch: MSRCR:=r, r=G[MSRCR-b], then doing linear stretch. In practice, it
       doesn't work well, so we take another measure: Simplest Color Balance"""
    alpha = 125
    img = img.astype('double') + 1  #
    csum_log = np.log(np.sum(img, axis=2))
    msr = MultiScaleRetinex(img - 1, sigmas)  # -1
    r = (np.log(alpha * img) - csum_log[..., None]) * msr
    # beta=46;G=192;b=-30;r=G*(beta*r-b) #deprecated
    # mmin,mmax=np.min(r),np.max(r)
    # stretch=(r-mmin)/(mmax-mmin)*255 #linear stretch is unsatisfactory
    for i in range(r.shape[-1]):
        r[..., i] = simplest_color_balance(r[..., i], 0.01, 0.01)
    return r.astype('uint8')


### Multi-Scale Retinex with Chromaticity Preservation, see[4] Algorithm 2 in section 4
@measure_time
def retinex_MSRCP(img, sigmas=[12, 80, 250], s1=0.01, s2=0.01):
    """compare to others, simple and very fast"""
    Int = np.sum(img, axis=2) / 3
    Diffs = []
    for sigma in sigmas:
        Diffs.append(np.log(Int + 1) - np.log(gauss_blur(Int, sigma) + 1))
    MSR = sum(Diffs) / 3
    Int1 = simplest_color_balance(MSR, s1, s2)
    B = np.max(img, axis=2)
    A = np.min(np.stack((255 / (B + eps), Int1 / (Int + eps)), axis=2), axis=-1)
    return (A[..., None] * img).astype('uint8')


@measure_time
def retinex_AMSR(img, sigmas=[12, 80, 250]):
    """see Proposed Method ii in "An automated multi Scale Retinex with Color
       Restoration for image enhancement"(doi: 10.1109/NCC.2012.6176791)"""
    img = img.astype('double') + 1  #
    msr = MultiScaleRetinex(img - 1, sigmas, flag=False)  #
    y = 0.05
    for i in range(msr.shape[-1]):
        v, c = np.unique((msr[..., i] * 100).astype('int'), return_counts=True)
        sort_v_index = np.argsort(v)
        sort_v, sort_c = v[sort_v_index], c[sort_v_index]  # plot hist
        zero_ind = np.where(sort_v == 0)[0][0]
        zero_c = sort_c[zero_ind]
        #
        _ = np.where(sort_c[:zero_ind] <= zero_c * y)[0]
        if len(_) == 0:
            low_ind = 0
        else:
            low_ind = _[-1]
        _ = np.where(sort_c[zero_ind + 1:] <= zero_c * y)[0]
        if len(_) == 0:
            up_ind = len(sort_c) - 1
        else:
            up_ind = _[0] + zero_ind + 1
        #
        low_v, up_v = sort_v[[low_ind, up_ind]] / 100  # low clip value and up clip value
        msr[..., i] = np.maximum(np.minimum(msr[:, :, i], up_v), low_v)
        mmin = np.min(msr[..., i])
        mmax = np.max(msr[..., i])
        msr[..., i] = (msr[..., i] - mmin) / (mmax - mmin) * 255
    msr = msr.astype('uint8')
    return msr


if __name__ == '__main__':
    dir_path = r"../../data"
    img = get_single_image(dir_path)
    retinex_SSR_img = retinex_SSR(img, 15)
    cv2.imshow('original_image', img)
    cv2.imshow('retinex_SSR', retinex_SSR_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

