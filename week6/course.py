import cv2
import numpy as np

def main():
    img = cv2.imread('airport_gray.tif', cv2.IMREAD_GRAYSCALE)
    #airport_gray.tif

    #Roberts
    #Roberts(img)

    #Prewitt
    # Prewitt(img)

    #Sobel
    # Sobel(img)

    #LaplacianFilter
    LaplacianFilter(img)
    LaplacianFilter(img, True)

    #orientalFilter
    #orientalFilter(img,135,0)

def Roberts(src):
    srcori = src
    # 扩充图像
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float64(src)

    # Roberts滤波
    rows,cols = src.shape
    gx = np.zeros(src.shape, dtype=np.float32)
    gy = np.zeros(src.shape, dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1,cols-1):
            gx[i,j] = src[i+1,j+1]-src[i,j]
            gy[i,j] = src[i+1,j]-src[i,j+1]
    g = np.abs(gx)+np.abs(gy)

    # 最小-最大值拉伸
    gmin = np.min(g)
    gmax = np.max(g)
    g = 255.0 / (gmax-gmin) * (g-gmin)
    g = np.uint8(g + 0.5)

    retral, g = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY)

    # 裁剪结果，并返回结果
    dst = g[1: rows-1, 1: cols-1]

    # show images
    cv2.namedWindow('input', 0)
    cv2.imshow('input', srcori)
    cv2.namedWindow('output', 0)
    cv2.imshow('output', dst)
    cv2.imwrite('buildings_roberts.tif',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Prewitt(src):
    srcori = src
    # 扩充图像
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float64(src)

    # 定义两个方向梯度变量、两个方向梯度计算模板
    gx = np.zeros(src.shape, dtype = np.float32)
    gy = np.zeros(src.shape, dtype = np.float32)
    tempx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype = np.float32)
    tempy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype = np.float32)

    # 卷积运算
    rows, cols = src.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            temp = src[i-1: i+2, j-1: j+2]
            gx[i, j] = np.sum(temp * tempx)
            gy[i, j] = np.sum(temp * tempy)
    g = np.abs(gx) + np.abs(gy)

    # 最小-最大值拉伸
    gmin = np.min(g)
    gmax = np.max(g)
    g = 255.0 / (gmax-gmin) * (g-gmin)
    g = np.uint8(g + 0.5)

    retral, g = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY)

    # 裁剪结果，并返回结果
    dst = g[1: rows-1, 1: cols-1]
    cv2.namedWindow('input', 0)
    cv2.imshow('input', srcori)
    cv2.namedWindow('output', 0)
    cv2.imshow('output', dst)
    cv2.imwrite('prewitt.tif',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Sobel(src):
    srcori = src
    # 扩充图像
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float32(src)

    # 定义梯度变量与滤波模板
    gx = np.zeros(src.shape, dtype = np.float32)
    gy = np.zeros(src.shape, dtype = np.float32)
    tempx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    tempy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    # Sobel滤波
    rows, cols = src.shape[0: 2]
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            temp = src[i-1: i+2, j-1: j+2]
            gx[i, j] = np.sum(temp * tempx)
            gy[i, j] = np.sum(temp * tempy)
    g = np.abs(gx) + np.abs(gy)

    # 最小-最大值拉伸
    gmin = np.min(g)
    gmax = np.max(g)
    g = 255.0 / (gmax-gmin) * (g-gmin)
    g = np.uint8(np.round(g))

    retral, g = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY)

    # 裁剪结果，并返回结果
    dst = g[1: rows-1, 1: cols-1]

    cv2.namedWindow('input', 0)
    cv2.imshow('input', srcori)
    cv2.namedWindow('output', 0)
    cv2.imshow('output', dst)
    cv2.imwrite('sobel.tif',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def LaplacianFilter(src, background=False):
    srcori = src
    # 为图像添加边框，并将图像转换成np.float64类型
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float64(src)

    # 定义滤波器(True为有背景，False为无背景)
    g = np.zeros(src.shape, dtype=np.float64)
    # if background:
    template = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    # else:
    #     template = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    # Laplacian滤波
    rows, cols = src.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            temp = src[i - 1: i + 2, j - 1: j + 2]
            g[i, j] = np.sum(temp * template)
    g = np.abs(g)

    if background:
        # 梯度的值可能会超出255，使用最小-最大值将结果拉伸到[0, 255]
        # gmin = np.min(g)
        # gmax = np.max(g)
        # g = 255.0 / (gmax - gmin) * (g - gmin)
        # g = np.uint8(np.round(g))
        # retral, g = cv2.threshold(g, 25, 100, cv2.THRESH_BINARY)
        g = src + g
        gmin = np.min(g)
        gmax = np.max(g)
        g = 255.0 / (gmax - gmin) * (g - gmin)
        g = np.uint8(np.round(g))
    else:
        # 梯度的值可能会超出255，使用最小-最大值将结果拉伸到[0, 255]
        gmin = np.min(g)
        gmax = np.max(g)
        g = 255.0 / (gmax - gmin) * (g - gmin)
        g = np.uint8(np.round(g))
        # retral, g = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY)

    # 裁剪滤波结果，并返回滤波结果
    dst = g[1: rows - 1, 1: cols - 1]

    cv2.namedWindow('input', 0)
    cv2.imshow('input', srcori)
    cv2.namedWindow('output', 0)
    cv2.imshow('output', dst)
    cv2.imwrite('LaplacianFilter.tif', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def orientalFilter(src, orientation, center):
    srcori = src
    # 扩充图像
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float32(src)

    # 定义滤波模板
    #水平方向
    if orientation == 0:
        if center == 0:
            template = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        elif center == 2:
            template = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    #45度方向
    elif orientation == 45:
        if center == 0:
            template = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=np.float32)
        elif center == 2:
            template = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
    #垂直方向
    elif orientation == 90:
        if center == 0:
            template = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        elif center == 2:
            template = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
    #135度方向
    elif orientation == 135:
        if center == 0:
            template = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=np.float32)
        elif center == 2:
            template = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

    # 方向滤波
    rows, cols = src.shape[0: 2]
    g = np.zeros(src.shape[0: 2], dtype=np.float32)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            temp = src[i - 1: i + 2, j - 1: j + 2]
            g[i, j] = np.sum(temp * template)
    g = np.abs(g)

    # 最小-最大值滤波
    gmin = np.min(g)
    gmax = np.max(g)
    g = 255.0 / (gmax - gmin) * (g - gmin)
    g = np.uint8(g + 0.5)

    retral, g = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY)

    # 裁剪并返回结果
    dst = g[1: rows - 1, 1: cols - 1]

    cv2.namedWindow('input', 0)
    cv2.imshow('input', srcori)
    cv2.namedWindow('output'+str(orientation)+str(center), 0)
    cv2.imshow('output'+str(orientation)+str(center), dst)
    cv2.imwrite('orientalFilter.tif', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()

