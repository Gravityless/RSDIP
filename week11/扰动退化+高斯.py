import cv2
import numpy as np
import matplotlib.pyplot as plt

# oriImg, original image
# degImg, degradated image

# step 1. load image
oriImg = cv2.imread('NJU.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = oriImg.shape

# step 2.1 degradating - fft
orif = np.fft.fft2(oriImg)
orifsh = np.fft.fftshift(orif)

# step 2.2 degradating - template
k = 0.0025
m = round(rows/2)
n = round(cols/2)
template = np.zeros((rows, cols), dtype = np.float64)
for i in range(rows):
    for j in range(cols):
        dist2 = (i-m)**2 + (j-n)**2
        template[i, j] = np.exp(-k * dist2**(5.0/6.0))

# step 2.3 degradating - filtering
degfsh = template * orifsh
degf = np.fft.ifftshift(degfsh)
degImg = np.fft.ifft2(degf)
degImg = np.abs(degImg)

# step 2.4 add noise
GaussNoise = np.random.normal(0, 0.1, (rows, cols))
degNoiseImg = degImg + GaussNoise*128
degNoiseImg[degNoiseImg<0] = 0
degNoiseImg[degNoiseImg>255] = 255
img2 = degNoiseImg

# step 3.1 fft
f2 = np.fft.fft2(img2)
fsh2 = np.fft.fftshift(f2)

# step 3.2 inverse filter direct
dirReImgfsh = fsh2 / template
dirReImgf = np.fft.ifftshift(dirReImgfsh)
dirReImg = np.fft.ifft2(dirReImgf)
dirReImg = np.abs(dirReImg)

# step 4.1 wiener template
k = 0.005
temp = (np.abs(template))**2
wtemplate = temp / (template * (temp+k) + 0.00000001)

# step 4.2 wiener filtering
wfsh = fsh2 * wtemplate

# step 4.3 ifft
wf = np.fft.ifftshift(wfsh)
wImg = np.fft.ifft2(wf)
wImg = np.abs(wImg)

plt.subplot(2, 2, 1), plt.imshow(degImg, 'gray'), plt.title('degradation'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(degNoiseImg, 'gray'), plt.title('Gaussian Noise'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(dirReImg, 'gray'), plt.title('inverse filtering direct'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(wImg, 'gray'), plt.title('wiener filtering'), plt.xticks([]), plt.yticks([])
plt.show()
