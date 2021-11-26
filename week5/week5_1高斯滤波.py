import math
import numpy as np
import cv2


def main():
    origin_img = cv2.imread('NJU_noise.tif', -1)
    cv2.namedWindow('original image', 0)
    cv2.namedWindow('my filter', 0)
    cv2.namedWindow('built-in filter', 0)

    # 设置模板大小和参数sigma
    fsize = 5
    sigma = 1.1

    # 分别使用自己实现方法和内建方法进行高斯滤波
    adjust_img = gaussianFilter(origin_img, fsize, sigma)
    built_in_filter = cv2.GaussianBlur(origin_img, (fsize, fsize), sigma)

    cv2.imshow('original image', origin_img)
    cv2.imshow('my filter', adjust_img)
    cv2.imshow('built-in filter', built_in_filter)
    cv2.waitKey(0)

    cv2.imwrite('gaussianfilter 5x5.tif', adjust_img)


# 制作高斯模板
def gaussianTemplate(fsize, sigma):
    center = (fsize - 1) / 2
    template = np.zeros([fsize, fsize], dtype=np.float32)
    for i in range(fsize):
        x = pow(i - center, 2)
        for j in range(fsize):
            y = pow(j - center, 2)
            template[i][j] = 1.0 / (2 * math.pi * sigma * sigma) * math.exp(-(x + y) / (2.0 * sigma * sigma))

    k = 1 / template[0][0]
    template = np.uint8(template * k + 0.5)
    return template


# 高斯滤波
def gaussianFilter(src, fsize=3, sigma=0.8):
    template = gaussianTemplate(fsize, sigma)
    border = np.uint8((fsize - 1) / 2)

    src = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_REPLICATE)
    dst = np.zeros(src.shape, dtype=np.float32)
    rows, cols, dim = src.shape
    for i in range(border, rows - border):
        for j in range(border, cols - border):
            for k in range(dim):
                temp = src[i - border: i + border + 1, j - border: j + border + 1, k]
                dst[i, j, k] = np.sum(1.0 * temp * template / np.sum(template))

    dst = np.uint8(dst + 0.5)
    dst = dst[border: rows - border, border: cols - border, :]
    return dst


if __name__ == '__main__':
    main()
