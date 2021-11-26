import numpy as np
import cv2
import matplotlib.pyplot as plt

def statHist(src):
    rows = src.shape[0]
    cols = src.shape[1]
    # 定义1*256矩阵，存储直方图统计量
    hist = np.zeros(256, dtype = np.float32)

    # 遍历像元值，统计每个DN值出现频次
    for i in range(rows):
            for j in range(cols):
                dn = src[i, j]
                hist[dn] += 1

    # 绘制直方图，整饰
    # 产生0.5的偏移量，使数值正好在条形的正中央
    DN = np.arange(256) - 0.5
    # width = 1 不可少，否则是条形图而不是直方图
    plt.bar(DN, hist, width = 1)
    # 产生0.5的偏移量，使数值正好在条形的正中央
    plt.xlim([0-0.5, 255.5])

    plt.show()
    return

img = cv2.imread('NJU_gray.tif')
plt.imshow(img, cmap='gray')
plt.show()
statHist(img)