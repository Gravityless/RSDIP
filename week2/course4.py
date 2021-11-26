import matplotlib.pyplot as plt
import cv2

img = cv2.imread('test02img.jpg', 1)

histb = cv2.calcHist([img], [0], None, [256], [0, 255])
histg = cv2.calcHist([img], [1], None, [256], [0, 255])
histr = cv2.calcHist([img], [2], None, [256], [0, 255])

cv2.waitKey(0)
plt.plot(histb, 'b')
plt.plot(histg, 'g')
plt.plot(histr, 'r')
plt.show()

#对第1子图进行设定
plt.subplot(3, 1, 1)
plt.plot(histb, 'b')
plt.xlabel('x')
plt.ylabel('y')
#对第2子图进行设定
plt.subplot(3, 1, 2)
plt.plot(histg, 'g')
plt.xlabel('x')
plt.ylabel('y')
#对第3子图进行设定
plt.subplot(3, 1, 3)
plt.plot(histr, 'r')
plt.xlabel('x')
plt.ylabel('y')
#显示图像
plt.show()


