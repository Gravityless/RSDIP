import numpy as np
import cv2

def meanFilter(src):
    # 扩充一个边缘，宽度为1
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    dst = np.zeros(src.shape, dtype = np.float32) # 均值滤波结果

    # 均值滤波
    rows, cols, dim = src.shape
    template = 1.0/9 * np.ones((3, 3), dtype = np.float32)           # 生成均值滤波模板
    for i in range(1, rows-1):
        for j in range(1, cols-1):
             for k in range(dim):
                  temp = src[i-1: i+2, j-1: j+2, k]           # 取出参与运算的像元值
                  dst[i, j, k] = np.sum(temp * template)      # 当前像元和邻域与模板相乘，再求和
    dst = np.uint8(dst + 0.5)

    # 裁剪出最终结果
    dst = dst[1: rows-1, 1: cols-1,:]
    return dst
