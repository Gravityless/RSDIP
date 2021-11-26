import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalize(img):
    img = np.float32(img)
    max = np.max(img)
    min = np.min(img)
    img = 255/(max - min)*(img - min)
    img = np.uint8(img + 0.5)
    return img

def showFunc(temp_b,temp_g,temp_r):
    rows, cols = temp_b.shape
    m = np.int32(np.round(rows/2))
    n = np.int32(np.round(cols/2))
    fig = plt.figure()
    y = temp_b[m, n:]
    x = range(len(y))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.set_xlim([0, len(y)])
    ax.set_ylim([-0.1, 2.5])
    ax.set_xticks(np.linspace(0, len(y), 5))
    ax.set_yticks(np.linspace(0, 2.5, 6))
    ax.plot(x, y, linewidth=2, color='b')
    ax.plot(x, temp_g[m, n:], linewidth=2, color='g')
    ax.plot(x, temp_r[m, n:], linewidth=2, color='r')
    plt.show()
    return

def HF(src, H, L, d0, c):
    f = np.fft.fft2(src)
    fshift = np.fft.fftshift(f)

    rows, cols = src.shape
    m, n = rows/2, cols/2
    d0 = np.float32(d0)
    temp = np.zeros(src.shape, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i-m)**2 + (j-n)**2)
            temp[i, j] = 1 - np.exp(-c*d**2/d0**2)
    template = (H - L)*temp + L

    fshift2 = fshift*template
    f2 = np.fft.ifftshift(fshift2)
    dst = np.fft.ifft2(f2)
    dst = np.abs(dst)
    dst = normalize(dst)
    return dst, template

img = cv2.imread('homework.tif')
b, g, r = cv2.split(img)
dst_b, temp_b = HF(b, H=2, L=0.55, d0=30, c=2)
dst_g, temp_g = HF(g, H=2, L=0.6, d0=15, c=2)
dst_r, temp_r = HF(r, H=2, L=0.8, d0=10, c=2)
showFunc(temp_b,temp_g,temp_r)
merge_img = cv2.merge([dst_b, dst_g, dst_r])

cv2.imshow('output', merge_img)
cv2.waitKey(0)
cv2.imwrite('output.tif', merge_img)