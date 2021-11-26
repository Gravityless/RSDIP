import cv2
import numpy as np

def thresholdImage(src, thres):
    src[src > thres] = 255
    src[src <= thres] = 0
    dst = src
    return dst

img = cv2.imread(r'NJU_gray_1.tif', 0)
cv2.imshow('input', img)

img2 = thresholdImage(img, 70)
cv2.imshow('output', img2)
#cv2.imwrite(r'NJU_threshold_out.jpg', img2)
cv2.waitKey()
cv2.destroyAllWindows()