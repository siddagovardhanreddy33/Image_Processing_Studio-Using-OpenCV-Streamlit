import cv2
import numpy as np


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def blur(img):
    return cv2.GaussianBlur(img, (5,5), 0)


def threshold(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    return thresh


def negative(img):
    return cv2.bitwise_not(img)

def sepia(img):
    kernel = np.array([
        [0.272,0.534,0.131],
        [0.349,0.686,0.168],
        [0.393,0.769,0.189]
    ])

    return cv2.transform(img,kernel)