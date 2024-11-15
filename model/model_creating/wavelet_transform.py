import numpy as np
import pywt # pywavelets libary for wavelet transformations
import cv2

def w2d(img, mode='haar', level=1):
    """
    Apply 2D Discrete Wavelet Transformation (DWT) to an image

    The wavelet transformation helps to highlight the smooth, edges, textures, and sharp transitions in a facial image.
    """

    # convert image to grayscale
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # convert image to np array and normalize to avoid overflow and keep consistency
    imArray = np.float32(imArray) / 255

    # perform the wavelet decomposition
    coeffs = pywt.wavedec2(imArray, mode, level=level) # level specifys what level to decompose

    # set approximation coefficient (low frequency components) to 0
        # helps focis on deatails
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # remake the image with the modified coefficients
    imArray_H = pywt.waverec2(coeffs_H, mode)

    # rescale image back to 0-255 for visualization and convert to uint 8 for format compatability
    imArray_H = np.uint8(imArray_H * 255)

    # return modified image
    return imArray_H
