import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#uniform out-of-focus blur
def focus_blur(src, R):
    rows, cols = src.shape

    src_fft = np.fft.fft2(src)
    src_sh = np.fft.fftshift(src_fft)

    m = round(rows/2)
    n = round(cols/2)
    r2 = R*R
    temp = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            dist2 = (i-m)**2 + (j-n)**2
            if dist2 <= r2:
                temp[i, j] = 1/(math.pi * r2 + 0.000001)

    dg_sh = temp * src_sh
    dg_fft = np.fft.ifftshift(dg_sh)
    dg_img = np.abs(np.fft.ifft2(dg_fft))

    return dg_img, temp

#gaussian noise
def gaussian_noise(src):
    rows, cols = src.shape

    gaussNoise = np.random.normal(0, 0.1, (rows, cols))

    src = strench(src)

    degNoiseImg = src + gaussNoise * 128
    degNoiseImg[degNoiseImg < 0] = 0
    degNoiseImg[degNoiseImg > 255] = 255

    return degNoiseImg

#inverse filter
def inverse_filter(src, temp):
    src_fft = np.fft.fft2(src)
    src_sh = np.fft.fftshift(src_fft)

    inv_sh = src_sh/(temp + 0.000001)
    inv_img = np.abs(np.fft.ifft2(np.fft.ifftshift(inv_sh)))

    return inv_img


#wiener filter
def wiener_filter(src, template, k):
    f2 = np.fft.fft2(src)
    fsh2 = np.fft.fftshift(f2)

    temp = (np.abs(template)) ** 2
    wtemp = temp / (template * (temp + k) + 0.000001)

    # wtemp = template / (template**2 + k)

    wfsh = fsh2 * wtemp

    wf = np.fft.ifftshift(wfsh)
    wImg = np.fft.ifft2(wf)
    wImg = np.abs(wImg)

    return wImg

def strench(src):
    max = np.max(src)
    min = np.min(src)
    dst = (src - min)/(max - min) * 255

    return dst

def main():
    img = cv2.imread('NJU.tif', cv2.IMREAD_GRAYSCALE)
    img_blur, temp = focus_blur(img, 50)
    img_noise = gaussian_noise(img_blur)
    # img_noise = img_blur
    img_inv = inverse_filter(img_noise, temp)
    img_wn = wiener_filter(img_noise, temp, 0.01)

    plt.subplot(2, 2, 1), plt.imshow(img_blur, 'gray'), plt.title('degradation'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(img_noise, 'gray'), plt.title('Gaussian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(img_inv, 'gray'), plt.title('Inverse'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_wn, 'gray'), plt.title('Wiener'), plt.xticks([]), plt.yticks([])
    plt.show()



if __name__ == '__main__':
    main()

