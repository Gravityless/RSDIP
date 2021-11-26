import numpy as np
import cv2

# 最大值滤波(待滤波图像，滤波窗口大小)
def maxFilter(src, fsize = 3):
    # 为图像添加边缘，宽度为滤波窗口大小减一的一半
    border = np.uint8((fsize - 1) / 2)
    src = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_REPLICATE)

    # 最大值滤波
    dst = np.zeros(src.shape, dtype = np.uint8) # 滤波结果变量
    rows, cols , dim = src.shape
    for i in range(border, rows-border):
        for j in range(border, cols-border):
             for k in range(dim):
                  temp = src[i-border: i+1+border, j-border: j+1+border, k]
# 取出滤波窗口对应的像元值
                  dst[i, j, k] = np.max(temp) # 取出最大值，并赋值给结果像元

    # 裁剪出结果，并返回结果
    dst = dst[border: rows-border, border: cols-border, :]
    return dst
