import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

def main():
    img = cv2.imread('city.tif', 0)
    cv2.namedWindow('image')
    cv2.namedWindow('adjusted_image  print \'space\' to show hist, \'q\' to escape')

    #设置γ系数和常数c进度条
    cv2.createTrackbar('r*10^(-2)', 'adjusted_image  print \'space\' to show hist, \'q\' to escape', 500, 1000, nothing)
    cv2.createTrackbar('c*10^(-1)', 'adjusted_image  print \'space\' to show hist, \'q\' to escape', 5, 10, nothing)

    #获取γ值和c值
    gamma = cv2.getTrackbarPos('r*10^(-2)', 'adjusted_image  print \'space\' to show hist, \'q\' to escape') * 10 ** (-2)
    c = cv2.getTrackbarPos('c*10^(-1)', 'adjusted_image  print \'space\' to show hist, \'q\' to escape') * 0.1

    print('拖动进度条改变γ值和常数c值，\n按‘空格键’绘制当前图像直方图，按’q‘退出')

    while (1) :
        #根据参数创建查找表
        lut = np.array([(c * (i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
        lut[lut > 255] = 255
        lut[lut < 0] = 0
        dst = cv2.LUT(img, lut)

        cv2.imshow('image', img)
        cv2.imshow('adjusted_image  print \'space\' to show hist, \'q\' to escape', dst)
        k = cv2.waitKey(1)

        #根据键盘输入进行操作
        if(k == ord(' ')):
            hist1 = cv2.calcHist([img], [0], None, [256], [0, 255])
            hist2 = cv2.calcHist([dst], [0], None, [256], [0, 255])

            plt.subplot(2, 1, 1)
            plt.plot(hist1, 'y', label='Before')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(hist2, 'c', label='After')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()
        if(k == ord('q')):
            break

        #获取γ值和c值
        gamma = cv2.getTrackbarPos('r*10^(-2)', 'adjusted_image  print \'space\' to show hist, \'q\' to escape') * 10**(-2)
        c = cv2.getTrackbarPos('c*10^(-1)', 'adjusted_image  print \'space\' to show hist, \'q\' to escape') * 0.1

if __name__=='__main__':
    main()