import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    """
    Apply wavelet transformation
    """

    imArray = img

    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray / 255) #convert to float
    coeffs=pywt.wavedec2(imArray, mode, level=level) # perform 2d wavelet decomposition

    # set approximation coefficient (low frequency components) to 0
        # helps focis on deatails
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H =  np.uint8(imArray_H * 255)

    return imArray_H