import cv2
import numpy as np
import matplotlib.pyplot as plt

#对数变换
def logStretch(src, a, b):
    src = np.float32(src)
    dst = a * np.log(1 + src) + b # 四舍五入
    dst = np.uint8(dst + 0.5) # 强制转到 [0-255]
    return dst

img = cv2.imread('NJU_gray.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('input', img)

img2 = logStretch(img, 125, 0.5) #设计一个函数使图像灰度值拉伸到 0-255
cv2.imshow('output', img2)
#cv2.imwrite('NJU_gray_2.tif', img2)
cv2.waitKey(0)

#计算直方图
histb = cv2.calcHist([img], [0], None, [256], [0, 255])
hista = cv2.calcHist([img2], [0], None, [256], [0, 255])

#对第 1 子图进行设定
plt.subplot(2, 1, 1)
plt.plot(histb, 'y', label='Before')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

#对第 2 子图进行设定
plt.subplot(2, 1, 2)
plt.plot(hista, 'c', label='After')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
cv2.destroyAllWindows()