import cv2
import numpy as np

def Brovey(mul, pan):
    mul = np.float64(mul)
    pan = np.float64(pan)
    dst = np.zeros(mul.shape, dtype=np.float64)

    row, col, dim = mul.shape
    for i in range(row):
        for j in range(col):
            for k in range(dim):
                dst[i, j, k] = (mul[i, j, k]/(np.sum(mul[i, j])+0.000001))*pan[i, j]

    # for i in range(dim):
    #     imax = np.max(dst[:, :, i])
    #     imin = np.min(dst[:, :, i])
    #     dst[:, :, i] = 255.0 / (imax - imin + 0.000001) * (dst[:, :, i] - imin)

    # bands = cv2.split(dst)
    # for i in range(dim):
    #     min = np.min(bands[i])
    #     max = np.max(bands[i])
    #     bands[i] = (bands[i] - min)/(max - min) * 255
    # dst = cv2.merge(bands)

    dst = np.uint8(dst + 0.5)
    for i in range(dim):
        dst[:, :, i] = cv2.equalizeHist(dst[:, :, i])

    return dst

mul = cv2.imread('mul_input.tif', cv2.IMREAD_COLOR)
pan = cv2.imread('pan_input.tif', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('mul-image', mul)
# cv2.imshow('pan-image', pan)

img = Brovey(mul, pan)
cv2.imshow('fused image', img)

cv2.waitKey()
cv2.destroyAllWindows()