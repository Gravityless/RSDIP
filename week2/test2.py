import cv2
import numpy as np

img = cv2.imread('test02img.jpg', 1)

img_arr = np.array(img)
img_f_arr = np.swapaxes(img_arr, 0, 1)

print(img_arr)
print(img_arr.shape)

cv2.namedWindow('test2', 0)
cv2.namedWindow('test2_f', 0)

cv2.imshow('test2', img_arr)
cv2.imshow('test2_f', img_f_arr)

cv2.waitKey(0)