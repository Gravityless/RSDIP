import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义函数，统计和绘制图像直方图和图像累积直方图
def histPlot(src):
    row = src.shape[0]
    col = src.shape[1]

    # 统计图像直方图
    hist = np.zeros(256, dtype=np.float32)
    for i in range(row):
        for j in range(col):
            index = src[i, j]
            hist[index] += 1

    # 统计图像累积直方图
    cumHist = np.zeros(256, dtype=np.float32)
    cumHist[0] = hist[0]
    for i in range(1, 256):
        cumHist[i] = cumHist[i - 1] + hist[i]

    DN = range(256)
	#双Y轴展示直方图和积累图
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(DN, hist, color='b', linewidth=2, label = 'Histogram')
    ax1.set_ylabel('Histogram')

    ax2 = ax1.twinx()  # important function
    ax2.plot(DN, cumHist, color='r', linewidth = 2, label = 'Cumulative Histogram')
    ax2.set_ylabel('Cumulative Histogram')
	#整饰
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform = ax1.transAxes)
    fig.tight_layout()
    plt.xlim([0, 256])
    plt.show()
    return cumHist

# 定义函数，实现图像的直方图均衡化
def histEqual(src):
    row = src.shape[0]
    col = src.shape[1]

    cumHist = histPlot(src)

    # 生成查找表，利用查找表进行图像直方图均衡化
    lut = np.zeros(256, dtype=np.float32)
    pixelnum = row * col
    for i in range(256):
        lut[i] = 255.0 / pixelnum * cumHist[i]
    lut = np.uint8(lut + 0.5)
    dst = cv2.LUT(src, lut)  # 查找表变换

    return dst

img = cv2.imread('NJU_gray.tif')
dst = histEqual(img)
histPlot(dst)