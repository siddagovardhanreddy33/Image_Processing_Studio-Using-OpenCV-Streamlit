import cv2
import numpy as np


# Brightness and Contrast
def brightness_contrast(img, brightness, contrast):
    output = cv2.convertScaleAbs(img,alpha=contrast,beta=brightness)
    return output


# Sharpening
def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    output = cv2.filter2D(img,-1,kernel)
    return output