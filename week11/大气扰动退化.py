import cv2
import numpy as np
import matplotlib.pyplot as plt



oriImg = cv2.imread('NJU.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = oriImg.shape
orif = np.fft.fft2(oriImg)
orifsh = np.fft.fftshift(orif)
#构造大气湍流模型
k = 0.0025
m = round(rows/2)
n = round(cols/2)
template = np.zeros((rows, cols), dtype = np.float64)
for i in range(rows):
    for j in range(cols):
        dist2 = (i-m)**2 + (j-n)**2
        template[i, j] = np.exp(-k * dist2**(5.0/6.0))
#图像退化处理
degfsh = template * orifsh
#傅里叶逆变换
degf = np.fft.ifftshift(degfsh)
degImg = np.fft.ifft2(degf)
degImg = np.abs(degImg)

#直接逆滤波
degf2 = np.fft.fft2(degImg)
degfsh2 = np.fft.fftshift(degf2)
dirReImgfsh = degfsh2 / (template+0.00000001)
dirReImgf = np.fft.ifftshift(dirReImgfsh)
dirReImg = np.fft.ifft2(dirReImgf)
dirReImg = np.abs(dirReImg)
plt.subplot(2, 2, 3)
plt.imshow(dirReImg, 'gray')
plt.title('inverse filtering direct')
plt.xticks([])
plt.yticks([])


#Wiener滤波
k2 = 0.000005
temp = (np.abs(template))**2
wtemplate = temp / (template * (temp+k2) + 0.00000001)
wfsh = degfsh2 * wtemplate
wf = np.fft.ifftshift(wfsh)
wtemp = np.fft.ifft2(wf)
wImg = np.abs(wtemp)
plt.subplot(2, 2, 4)
plt.imshow(wImg, 'gray')
plt.title('wiener filtering')
plt.xticks([])
plt.yticks([])

#显示图像
plt.subplot(2, 2, 1)
plt.imshow(oriImg, 'gray')
plt.title('original image')
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(degImg, 'gray')
plt.title('degradation')
plt.xticks([])
plt.yticks([])
plt.show()




