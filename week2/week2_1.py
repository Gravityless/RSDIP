import cv2
import numpy as np

img = cv2.imread('test02img.jpg', 1)

rows, cols, nums = img.shape

cv2.namedWindow('image', 0)

print('按下对称轴对应字母以对称(x/y)\n按下esc键退出')

while True:
    cv2.imshow('image', img)
    button = cv2.waitKey(0)

    if button == 27:
        cv2.destroyAllWindows()
        print('退出！')
        break

    if button == ord('x'):
        for i in range(rows//2):
            temp_arr = img[i].copy()
            img[i] = img[rows-1-i]
            img[rows-1-i] = temp_arr
        button = -1
        print('沿x轴对称！')

    if button == ord('y'):
        for i in range(cols//2):
            temp_arr = img[:, i].copy()
            img[:, i] = img[:, cols-1-i]
            img[:, cols-1-i] = temp_arr
        button = -1
        print('沿y轴对称！')

    if button != -1:
        print('input illegal!')
        button = -1