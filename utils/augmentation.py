import cv2
import numpy as np


def horizontal_flip(img):
    return cv2.flip(img,1)


def vertical_flip(img):
    return cv2.flip(img,0)


def rotate(img):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2),90,1)
    return cv2.warpAffine(img,matrix,(w, h))


def add_noise(img):
    noise = np.random.normal(0,25,img.shape)
    noisy = img + noise
    return np.clip(noisy,0,255).astype(np.uint8)