import cv2
import numpy as np
import matplotlib.pyplot as plt
#分段线性拉伸
def LDSPlot(x1, y1, x2, y2):
    plt.plot([0, x1, x2, 255], [0, y1, y2, 255], 'r' , label = ' Targetline', linewidth=2.5) #红色实线，定义 label，线宽
    plt.plot([0, 255], [0, 255], 'g' , label = ' Guideline', linewidth=2.5)
    plt.plot([0, x1, x1], [0, 0, x1], 'y--')
    plt.plot([0, x2, x2], [0, 0, y2], '--')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.legend() #显示图例
    plt.show()

def linearDividedStretch(src, x1, y1, x2, y2):
    # 以查找表的方式做分段线性变换
    LDSPlot(x1, y1, x2, y2)
    x1, x2, y1, y2 = np.float(x1), np.float(x2), np.float(y1), np.float(y2)

    x = np.arange(256) #列数为 256 的一维数组
    lut = np.zeros(256, dtype = np.float64) # 定义一个空表作为容器，储存每一灰度值对应变换后的值
    #填写内容
    for i in x:
        if(i < x1):
            lut[i] = (y1 * 1.0 / x1) * i
        elif(i < x2):
            lut[i] = (y2 - y1) / (x2 - x1) * (i - x1) + y1
        else:
            lut[i] = (255 - y2) / (255 - x2) * (i - x2) + y2
    lut = np.uint8(lut + 0.5)
    dst = cv2.LUT(src, lut)
    return dst

img = cv2.imread(r'city_gamma.tif', 0)
cv2.imshow('input', img)

img2 = linearDividedStretch(img, 20, 10, 200, 250)
cv2.imshow('output', img2)
#cv2.imwrite(r'city_out.tif', img2)
cv2.waitKey()

# 计算直方图计算直方图，比较拉伸前后直方，比较拉伸前后直方图变化图变化
histb = cv2.calcHist([img], [0], None, [256], [0, 255])
hista = cv2.calcHist([img2], [0], None, [256], [0, 255])

# 对第对第11子图进行设定子图进行设定
plt.subplot(2, 1, 1)
plt.plot(histb, 'y', label = 'Before')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# 对第对第22子图进行设定子图进行设定
plt.subplot(2, 1, 2)
plt.plot(hista, 'c', label = 'After')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
cv2.destroyAllWindows()
print('done!')