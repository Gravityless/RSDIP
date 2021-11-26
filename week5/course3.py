import numpy as np
import cv2

def addPepperSaltNoise(src, num):
    dst = src.copy() # 定义数组，存储结果

    # 添加椒盐噪声
    rows, cols = src.shape[0: 2]
    for k in range(num):
        # 生成随机数对(i, j)，作为噪声的位置
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)

        # 生成随机数0和1，决定是椒噪声还是盐噪声
        judge = np.random.randint(0, 2)

        # 灰度图像
        if src.ndim == 2:
            if judge == 0:
                dst[i, j] = 0
            else:
                dst[i, j] = 255

        # 彩色图像
        elif src.ndim == 3:
            if judge == 0:
                dst[i, j, :] = 0
            else:
                dst[i, j, :] = 255
    return dst
