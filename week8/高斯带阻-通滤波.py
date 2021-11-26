import cv2
import numpy as np
import matplotlib.pyplot as plt

def showTemplate(template, ftype):
    template = np.uint8(template * 255)
    cv2.imshow('template', template)
    cv2.imwrite(ftype + ' ' + 'template.tif',template)
    return

def showMag(fshift, name, ftype):
    mag = np.log(np.abs(fshift)+1)
    mmin = np.min(mag)
    mmax = np.max(mag)
    dst = 255.0 / (mmax - mmin) * (mag - mmin)
    cv2.imshow(name, np.uint8(dst + 0.5))
    cv2.imwrite(ftype + ' ' + name + '.tif',dst)
    return

def showFunction(template, ftype):
    rows, cols = template.shape
    m = np.uint8(rows / 2)
    n = np.uint8(cols / 2)
    y = template[m, n:]
    x = range(len(y))
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(x, y, color = 'b', linewidth = 2)
    ax.set_xlim([np.min(x),np.max(x)])
    ax.set_ylim([-0.1, 1.2])
    ax.set_xticks(np.linspace(0,n,5))
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_title(ftype + ' ' + 'Function')
    plt.savefig(ftype + ' ' + 'Function.png')
    plt.show()
    return

def GaussinBandFilter(src, d0, w, ftype = 'band-reject'):
    f = np.fft.fft2(src)
    fshift = np.fft.fftshift(f)
    showMag(fshift, 'original magnitude', ftype)

    template = np.zeros(src.shape, dtype = np.float32)
    rows, cols = src.shape
    x, y = rows/2, cols/2
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i-x)**2 + (j-y)**2)
            template[i, j] = 1 - (np.e)**(-0.5*((d**2-d0**2)/(d+0.000001)/w)**2)
    if ftype == 'band-pass':
        template = 1 - template
    showFunction(template, ftype)
    showTemplate(template, ftype)

    fshift2 = fshift * template
    showMag(fshift2, 'banded magnitude', ftype)

    dst = np.fft.ifft2(np.fft.ifftshift(fshift2))
    dst = abs(dst)
    dst[dst > 255] = 255
    dst[dst < 0] = 0
    return np.uint8(dst)

img = cv2.imread('car.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('input', img)

imgR = GaussinBandFilter(img, 50, 10, 'band-pass')
cv2.imshow('output band-pass', imgR)

cv2.waitKey(0)
cv2.destroyAllWindows()
