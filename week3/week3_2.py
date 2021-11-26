import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

def main():
    origin = cv2.imread('subset.tif', -1)
    print('拖动进度条使用分段线性变换，\n按‘空格键’绘制当前直方图和线性变换函数，按’q‘退出')
    cv2.namedWindow('original_image')
    cv2.namedWindow('adjusted_image')
    cv2.namedWindow('control_bars  print \'space\' to show charts, \'q\' to escape', 0)

    #线性拉伸以显示原图
    img = linearStretch(origin)[...,::-1]

    #初始化进度条
    initSettings()

    while (1) :
        #获取分段线性变换参数
        b1, b2, g1, g2, r1, r2, yb1, yb2, yg1, yg2, yr1, yr2 = getSettings()

        #分段线性变换
        ajt_img = LDS(img, b1, b2, g1, g2, r1, r2, yb1, yb2, yg1, yg2, yr1, yr2)
        cv2.imshow('original_image', img)
        cv2.imshow('adjusted_image', ajt_img)
        k = cv2.waitKey(1)

        if k == ord(' ') :

            #获取变换后直方图统计数据
            histb, histg, histr, hist2b, hist2g, hist2r = getHist(img, ajt_img)

            #绘制变换前后直方图和函数图像
            LDSPlot(histb,histg,histr,hist2b,hist2g,hist2r,b1, b2, g1, g2, r1, r2, yb1, yb2, yg1, yg2, yr1, yr2)

        if k == ord('q') :
            break

#获取变换后直方图统计数据
def getHist(img, ajt_img):
    histb = cv2.calcHist([img], [0], None, [256], [0, 255])
    histg = cv2.calcHist([img], [1], None, [256], [0, 255])
    histr = cv2.calcHist([img], [2], None, [256], [0, 255])
    hist2b = cv2.calcHist([ajt_img], [0], None, [256], [0, 255])
    hist2g = cv2.calcHist([ajt_img], [1], None, [256], [0, 255])
    hist2r = cv2.calcHist([ajt_img], [2], None, [256], [0, 255])
    return histb,histg,histr,hist2b,hist2g,hist2r

#绘制变换前后直方图和函数图像
def LDSPlot(histb,histg,histr,hist2b,hist2g,hist2r,b1, b2, g1, g2, r1, r2, yb1, yb2, yg1, yg2, yr1, yr2):
    plt.figure(figsize=[16,10],dpi=120)
    plt.subplot(3, 1, 1)
    plt.plot(histb, 'b', label='b')
    plt.plot(histg, 'g', label='g')
    plt.plot(histr, 'r', label='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('before')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(hist2b, 'b', label='b')
    plt.plot(hist2g, 'g', label='g')
    plt.plot(hist2r, 'r', label='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('after')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot([0, 255],[0, 255], 'k--')
    plt.plot([0, b1, b2, 255], [0, yb1, yb2, 255], 'b', label='b')
    plt.plot([0, g1, g2, 255], [0, yg1, yg2, 255], 'g', label='g')
    plt.plot([0, r1, r2, 255], [0, yr1, yr2, 255], 'r', label='r')
    plt.legend()
    plt.show()

#线性拉伸以显示原图
def linearStretch(src):
    smax = np.max(src)
    smin = np.min(src)
    img = 0.5 * 255.0 * (src - smin) / (smax - smin)
    img = np.uint8(img + 0.5)
    return img

#三通道查找表实现分段线性变换
def LDS(src, b1, b2, g1, g2, r1, r2, yb1, yb2, yg1, yg2, yr1, yr2):
    x = np.arange(256)
    lut1 = np.zeros(256, dtype = np.float64)
    lut2 = np.zeros(256, dtype=np.float64)
    lut3 = np.zeros(256, dtype=np.float64)

    for i in x:
        if i < b1 :
            lut1[i] = (yb1 * 1.0 / b1) * i
        elif i < b2 :
            lut1[i] = (yb2 - yb1) / (b2 - b1) * (i - b1) + yb1
        else :
            lut1[i] = (255 - yb2) / (255 - b2) * (i - b2) + yb2

    for i in x:
        if i < g1 :
            lut2[i] = (yg1 * 1.0 / g1) * i
        elif i < g2 :
            lut2[i] = (yg2 - yg1) / (g2 - g1) * (i - g1) + yg1
        else :
            lut2[i] = (255 - yg2) / (255 - g2) *(i - g2) + yg2

    for i in x:
        if i < r1 :
            lut3[i] = (yr1 * 1.0 / r1) * i
        elif i < r2 :
            lut3[i] = (yr2 - yr1) / (r2 - r1) * (i - r1) + yr1
        else :
            lut3[i] = (255 - yr2) / (255 - r2) *(i - r2) + yr2

    # for i in (lut1, lut2, lut3):
    #     i = np.uint8(i + 0.5)
    lut1 = np.uint8(lut1 + 0.5)
    lut2 = np.uint8(lut2 + 0.5)
    lut3 = np.uint8(lut3 + 0.5)
    lut_bgr = np.dstack((lut1,lut2,lut3))
    ajt_img = cv2.LUT(src, lut_bgr)

    return ajt_img

#获取进度条参数
def getSettings():
    b1 = cv2.getTrackbarPos('b1', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    g1 = cv2.getTrackbarPos('g1', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    r1 = cv2.getTrackbarPos('r1', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    b2 = cv2.getTrackbarPos('b2', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    g2 = cv2.getTrackbarPos('g2', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    r2 = cv2.getTrackbarPos('r2', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    yb1 = cv2.getTrackbarPos('yb1', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    yg1 = cv2.getTrackbarPos('yg1', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    yr1 = cv2.getTrackbarPos('yr1', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    yb2 = cv2.getTrackbarPos('yb2', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    yg2 = cv2.getTrackbarPos('yg2', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    yr2 = cv2.getTrackbarPos('yr2', 'control_bars  print \'space\' to show charts, \'q\' to escape')
    return b1, b2, g1, g2, r1, r2, yb1, yb2, yg1, yg2, yr1, yr2

#初始化进度条
def initSettings():
    cv2.createTrackbar('b1','control_bars  print \'space\' to show charts, \'q\' to escape', 20, 255, nothing)
    cv2.createTrackbar('g1','control_bars  print \'space\' to show charts, \'q\' to escape', 20, 255, nothing)
    cv2.createTrackbar('r1','control_bars  print \'space\' to show charts, \'q\' to escape', 20, 255, nothing)
    cv2.createTrackbar('yb1','control_bars  print \'space\' to show charts, \'q\' to escape', 55, 255, nothing)
    cv2.createTrackbar('yg1','control_bars  print \'space\' to show charts, \'q\' to escape', 55, 255, nothing)
    cv2.createTrackbar('yr1','control_bars  print \'space\' to show charts, \'q\' to escape', 55, 255, nothing)

    cv2.createTrackbar('b2','control_bars  print \'space\' to show charts, \'q\' to escape', 100, 255, nothing)
    cv2.createTrackbar('g2','control_bars  print \'space\' to show charts, \'q\' to escape', 80, 255, nothing)
    cv2.createTrackbar('r2','control_bars  print \'space\' to show charts, \'q\' to escape', 75, 255, nothing)
    cv2.createTrackbar('yb2','control_bars  print \'space\' to show charts, \'q\' to escape', 230, 255, nothing)
    cv2.createTrackbar('yg2','control_bars  print \'space\' to show charts, \'q\' to escape', 230, 255, nothing)
    cv2.createTrackbar('yr2','control_bars  print \'space\' to show charts, \'q\' to escape', 230, 255, nothing)

if __name__=='__main__':
    main()