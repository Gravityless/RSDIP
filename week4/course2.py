import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('NJU.png', -1)
img_gray = cv2.imread('NJU_gray.tif')
# 定义三个通道直方图的颜色
color = ['b', 'g', 'r']
for i, c in enumerate(color): # [(0, 'b'), (1, 'g'), (2, 'r')]
    hist = cv2.calcHist([img], [i], None, [256], [0, 255])
    plt.plot(hist, color = c)
    plt.xlim([0, 255])
plt.show()

n, bins, patches = plt.hist(img_gray.ravel(), 256, [0, 256])
plt.xlim(0, 255)
plt.show()

# 统计直方图
hist, bins2 = np.histogram(img_gray.flatten(), 50, [0, 256])
# 绘制直方图
plt.bar(bins2[:-1], hist, width = 2, color = 'b')
plt.xlim([0-0.5, 256-0.5])
plt.show()

