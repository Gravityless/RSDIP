import numpy as np
import cv2
import matplotlib.pyplot as plt

#Region Growing
def region_grow(src, seeds, thd):
    #将RGB转换为HSI空间
    hsi, bins = RGB2HSI(src)
    hsi_i = hsi[2].copy()
    #对I波段进行区域生长
    rows, cols = hsi_i.shape
    dst = np.zeros(hsi_i.shape, dtype=np.float64)
    orient = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], dtype = np.float64)
    growx = []
    growy = []
    for seed in seeds:
        growx.append(seed[0])
        growy.append(seed[1])
        check = np.zeros(hsi_i.shape)
        label = hsi_i[seed[0], seed[1]]
        dst[seed[0], seed[1]] = label
        size = 1
        mean_dst = hsi_i[seed[0], seed[1]]
        while(growx != []):
            curx = growx.pop(0)#当前种子点的行号
            cury = growy.pop(0)#当前种子点的列号
            sumtemp = 0
            count = 0
            for m in range(4):
                tempx = curx + orient[m][0]#邻域像素的行号
                tempy = cury + orient[m][1]#邻域像素的列号
                if tempx in range(rows) and tempy in range(cols):
                    tempx=np.uint16(tempx)
                    tempy=np.uint16(tempy)
                    dist = np.abs(hsi_i[tempx, tempy] - mean_dst)#邻域像素与种子点之间的差
                    if dist < thd and check[tempx, tempy] == 0:
                        check[tempx, tempy] = 1
                        dst[tempx, tempy] = label
                        growx.append(tempx)
                        growy.append(tempy)
                        sumtemp += hsi_i[tempx, tempy]
                        count = count + 1
            mean_dst = (mean_dst*size + np.float64(sumtemp)) / (size+count)
            size += count

    #将生长结果dst赋给HSI的I波段
    hsi[2] = dst
    for i in range(rows):
        for j in range(cols):
            if(hsi[2][i, j]==0):
                hsi[2][i, j]= hsi_i[i, j]

    #将HSI转换到RGB空间
    b, g, r = HSI2RGB(hsi, bins)

    dst = np.zeros(src.shape, dtype=np.uint8)
    dst[:, :, 0] = np.uint8(b)
    dst[:, :, 1] = np.uint8(g)
    dst[:, :, 2] = np.uint8(r)
    return dst

#RGB2HSI
def RGB2HSI(src):
    src = np.float64(src)
    r = src[:,:,2].copy()
    g = src[:,:,1].copy()
    b = src[:,:,0].copy()

    i = (r+g+b)/3.0
    v1 = np.sqrt(2) / 3.0 * b - np.sqrt(2) / 6.0 * g - np.sqrt(2) / 6.0 * r
    v2 = np.sqrt(2) / 2.0 * r - np.sqrt(2) / 2.0 * g
    h = np.arctan(v2 / (v1 + 0.0000001))
    s = np.sqrt(v1 * v1 + v2 * v2)

    hsi = [h,s,i]
    bins = [v1,v2]
    return hsi,bins

#HSI2RGB
def HSI2RGB(hsi, bins):
    i = hsi[2]
    v1 = bins[0]
    v2 = bins[1]

    r = i - np.sqrt(2) / 2.0 * v1 + np.sqrt(2) / 2.0 * v2
    g = i - np.sqrt(2) / 2.0 * v1 - np.sqrt(2) / 2.0 * v2
    b = i + np.sqrt(2) * v1

    return b,g,r


def main():
    img = cv2.imread('image.tif',cv2.IMREAD_COLOR)
    #高斯模糊
    img_gaussian = cv2.GaussianBlur(img,(3,3),0.73)
    #生长种子
    seeds=[(50,50),(50,200),(50,300),(50,400),(50,500),(100,100),(150,150),(200,200),(300,300),(400,400),(400,500),(450,500),(500,500)]
    # seeds = [(x, y) for x in range(512) if x%64==0 for y in range(512) if y%64==0]
    img_grow = region_grow(img_gaussian, seeds, thd=30)
    #显示图片
    cv2.imshow('input', img_gaussian)
    cv2.imshow('output', img_grow)
    cv2.waitKey(0)
    cv2.imwrite('output3.tif',img_grow)


if __name__=='__main__':
    main()