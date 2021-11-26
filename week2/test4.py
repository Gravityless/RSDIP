import cv2
import numpy as np

img = cv2.imread('test02img.jpg', 1)

cv2.namedWindow('img', 0)

#img = img[..., ::-1]

b,g,r=cv2.split(img)
img2=cv2.merge([r,g,b])

cv2.imshow('img', img2)

cv2.waitKey(0)
