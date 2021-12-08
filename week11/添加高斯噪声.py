import cv2
import numpy as np
import matplotlib.pyplot as plt
#
# step 1. load image
img = cv2.imread('NJU.tif', cv2.IMREAD_GRAYSCALE)
img2 = np.float64(img)


# step 2. add Gaussian Noise
rows, cols = img.shape
GaussNoise = np.random.normal(0, 0.1, (rows, cols))
img3 = img2 + GaussNoise*128


# step 3. postprocessing
img3[img3>255] = 255
img3[img3<0] = 0

# step 4. show image
plt.subplot(1, 2, 1), plt.imshow(img, 'gray'), plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(img3, 'gray'), plt.title('Gaussian'), plt.xticks([]), plt.yticks([])
plt.show()

#添加高斯噪声