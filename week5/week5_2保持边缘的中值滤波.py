import numpy as np
import cv2

def main():
    origin_img = cv2.imread('NJU_noise.tif', -1)
    cv2.namedWindow('original image', 0)
    cv2.namedWindow('hybrid median', 0)
    cv2.namedWindow('conventional median', 0)

    #设置模板大小
    fsize = 3

    #保持边缘的中值滤波和普通中值滤波
    adjust_img = medianWithBorder(origin_img, fsize)
    convention_img = medianFilter(origin_img, fsize)

    cv2.imshow('original image', origin_img)
    cv2.imshow('hybrid median', adjust_img)
    cv2.imshow('conventional median', convention_img)
    cv2.waitKey(0)

    cv2.imwrite('hybrid 3x3 median.tif', adjust_img)

#保持边缘的中值滤波
def medianWithBorder(src, fsize = 3):
    border = np.uint8((fsize - 1) / 2)

    src = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_REPLICATE)
    dst = np.zeros(src.shape, dtype=np.uint8)
    rows, cols = src.shape[0:2]

    #彩色图像处理
    if src.ndim == 3:
        for i in range(border, rows - border):
            for j in range(border, cols - border):
                for k in range(3):
                    temp1 = np.zeros(4 * border, dtype=np.uint8)
                    q = 0
                    for p in range(border):
                        temp1[q] = src[i - p - 1, j, k]
                        q += 1
                    for p in range(border):
                        temp1[q] = src[i + p + 1, j, k]
                        q += 1
                    for p in range(border):
                        temp1[q] = src[i, j - p - 1, k]
                        q += 1
                    for p in range(border):
                        temp1[q] = src[i, j + p + 1, k]
                        q += 1

                    temp2 = np.zeros(4 * border, dtype=np.uint8)
                    q = 0
                    for p in range(border):
                        temp2[q] = src[i - p - 1, j - p - 1, k]
                        q += 1
                    for p in range(border):
                        temp2[q] = src[i + p + 1, j - p - 1, k]
                        q += 1
                    for p in range(border):
                        temp2[q] = src[i - p - 1, j + p + 1, k]
                        q += 1
                    for p in range(border):
                        temp2[q] = src[i + p + 1, j + p + 1, k]
                        q += 1

                    median1 = np.median(temp1)
                    median2 = np.median(temp2)
                    dst[i, j, k] = np.median([median1, median2, src[i, j, k]])
        dst = dst[border: rows - border, border: cols - border, :]

    #灰度图像处理
    elif src.ndim == 2:
        for i in range(border, rows - border):
            for j in range(border, cols - border):
                temp1 = np.zeros(4 * border, dtype=np.uint8)
                q = 0
                for p in range(border):
                    temp1[q] = src[i - p - 1, j]
                    q += 1
                for p in range(border):
                    temp1[q] = src[i + p + 1, j]
                    q += 1
                for p in range(border):
                    temp1[q] = src[i, j - p - 1]
                    q += 1
                for p in range(border):
                    temp1[q] = src[i, j + p + 1]
                    q += 1

                temp2 = np.zeros(4 * border, dtype=np.uint8)
                q = 0
                for p in range(border):
                    temp2[q] = src[i - p - 1, j - p - 1]
                    q += 1
                for p in range(border):
                    temp2[q] = src[i + p + 1, j - p - 1]
                    q += 1
                for p in range(border):
                    temp2[q] = src[i - p - 1, j + p + 1]
                    q += 1
                for p in range(border):
                    temp2[q] = src[i + p + 1, j + p + 1]
                    q += 1

                median1 = np.median(temp1)
                median2 = np.median(temp2)
                dst[i, j] = np.median([median1, median2, src[i, j]])
        dst = dst[border: rows - border, border: cols - border]

    return dst

#普通中值滤波
def medianFilter(src, fsize = 3):
    border = np.uint8((fsize - 1) / 2)

    src = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_REPLICATE)
    dst = np.zeros(src.shape, dtype = np.uint8)
    rows, cols, dim = src.shape
    for i in range(border, rows - border):
        for j in range(border, cols - border):
             for k in range(dim):
                  temp = src[i - border: i + border + 1, j - border: j + border + 1, k]
                  dst[i, j, k] = np.median(temp)

    dst = dst[border : rows - border, border : cols - border, :]
    return dst


if __name__ == '__main__':
    main()