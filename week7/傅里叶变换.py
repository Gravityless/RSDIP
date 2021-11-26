import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    lena = cv2.imread('Lena.tif', 0)
    buildings = cv2.imread('buildings.jpg', 0)
    #傅里叶变换，低频居中
    lena_sp = np.fft.fftshift(np.fft.fft2(lena))
    build_sp = np.fft.fftshift(np.fft.fft2(buildings))
    #频谱
    lena_mag = np.abs(lena_sp)
    build_mag = np.abs(build_sp)
    #相位谱
    lena_pha = np.angle(lena_sp)
    build_pha = np.angle(build_sp)
    #lena频谱build相位
    lena_mag_build_pha = np.zeros(lena.shape, dtype = complex)
    lena_mag_build_pha.real = lena_mag * np.cos(build_pha)
    lena_mag_build_pha.imag = lena_mag * np.sin(build_pha)
    #build相位lena频谱
    build_mag_lena_pha = np.zeros(buildings.shape, dtype = complex)
    build_mag_lena_pha.real = build_mag * np.cos(lena_pha)
    build_mag_lena_pha.imag = build_mag * np.sin(lena_pha)
    #图像拉伸
    lena_mag_build_pha = min_max_strench(np.abs(np.fft.ifft2(np.fft.ifftshift(lena_mag_build_pha))))
    build_mag_lena_pha = min_max_strench(np.abs(np.fft.ifft2(np.fft.ifftshift(build_mag_lena_pha))))

    cv2.imshow('lena_mag_build_pha', lena_mag_build_pha)
    cv2.imshow('build_mag_lena_pha', build_mag_lena_pha)
    cv2.imwrite('lena_mag_build_pha.jpg', lena_mag_build_pha)
    cv2.imwrite('build_mag_lena_pha.jpg', build_mag_lena_pha)

    cv2.waitKey()
    cv2.destroyAllWindows()

def min_max_strench (magnitude):
    mmax = np.max(magnitude)
    mmin = np.min(magnitude)
    magnitude = 255.0 / (mmax - mmin) * (magnitude - mmin)
    return np.uint8(magnitude + 0.5)

if __name__ == '__main__':
    main()
