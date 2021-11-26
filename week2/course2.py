import cv2
import numpy as np

img = cv2.imread('Lena.jpg', 1)
cv2.imshow('Lena', img)

img_horizon = cv2.flip(img, 1)
cv2.imshow('horizon', img_horizon)

img_vertical = cv2.flip(img, 0)
cv2.imshow('vertical', img_vertical)

img_hzvt = cv2.flip(img, -1)
cv2.imshow('horizon_vertical', img_hzvt)

cv2.waitKey(0)