import cv2
import numpy as np

#利用IHS方法进行图像融合
def IHS(mul, pan):

    #将数据转换为浮点型，便于计算
    mul = np.float64(mul)
    pan = np.float64(pan)
    dst = np.zeros(mul.shape, dtype = np.uint8)

    #获取图像的RGB值
    #需要注意的是图像的波段顺序是按BGR依次排列
    r = mul[:, :, 2].copy()
    g = mul[:, :, 1].copy()
    b = mul[:, :, 0].copy()

    #计算相应的IHS
    i = (r + g + b) / 3.0
    v1 = np.sqrt(2) / 3.0 * b - np.sqrt(2) / 6.0 * g - np.sqrt(2) / 6.0 * r
    v2 = np.sqrt(2) / 2.0 * r - np.sqrt(2) / 2.0 * g
    h = np.arctan(v2 / (v1 + 0.0000001))
    s = np.sqrt(v1 * v1 + v2 * v2)

    #图像替换并转换到RGB空间
    #这里直接使用全色pan替换i，没有进行直方图匹配
    i = pan
    r = i - np.sqrt(2) / 2.0 * v1 + np.sqrt(2) / 2.0 * v2
    g = i - np.sqrt(2) / 2.0 * v1 - np.sqrt(2) / 2.0 * v2
    b = i + np.sqrt(2) * v1

    #对图像进行拉伸
    rmax = np.max(r)
    gmax = np.max(g)
    bmax = np.max(b)
    r = r / rmax * 255.0
    g = g / gmax * 255.0
    b = b / bmax * 255.0

    #将数据的值转换为整型
    dst[:, :, 0] = np.uint8(b)
    dst[:, :, 1] = np.uint8(g)
    dst[:, :, 2] = np.uint8(r)
    return dst

mul = cv2.imread('mul_input.tif', cv2.IMREAD_COLOR)
pan = cv2.imread('pan_input.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('mul', mul)
cv2.imshow('pan', pan)

img = IHS(mul, pan)
cv2.imshow('output', img)
# cv2.imwrite('output.png',img)

cv2.waitKey()
cv2.destroyAllWindows()
