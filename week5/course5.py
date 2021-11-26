import numpy as np
import cv2

#中值滤波
def medianFilter(src, fsize = 3):
    border = np.uint8((fsize - 1) / 2)

    src = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_REPLICATE)
    dst = np.zeros(src.shape, dtype = np.uint8)
    rows, cols, dim = src.shape
    for i in range(border, rows - border):
        for j in range(border, cols - border):
             for k in range(dim):
                  temp = src[i - border : i + border + 1, j - border : j + border + 1, k]
                  dst[i, j ,k] = np.median(temp)

    dst = dst[border : rows - border, border : cols - border, :]
    return dst

