import cv2

def edge_detect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray,100, 200)