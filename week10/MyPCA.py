import cv2
import numpy as np
import matplotlib.pyplot as plt

def myPCA(mul, pan):
    mul = np.float64(mul)
    pan = np.float64(pan)
    dst = np.zeros(mul.shape, dtype=np.uint8)

    # 步骤一 矩阵向量化 ravel()
    row, col, dim = mul.shape
    temp = np.zeros((row * col, dim), dtype = np.float32)
    for i in range(dim):
        temp[:, i] = (mul[:, :, i]).ravel()

    # 步骤二 求图像数组的协方差矩阵 np.dot()
    mean = np.mean(temp,axis=0)
    newData = temp - mean
    M = np.dot(newData.T, newData)/(row * col-1)

    # 步骤三 计算特征值和特征向量 np.linalg.eig()
    eigVals, eigVects = np.linalg.eig(M)

    # 步骤四 选择特征向量组成变换矩阵 np.argsort()
    eigIndice = np.argsort(eigVals)
    neig = eigIndice[-1:-dim-1:-1]
    nvec = eigVects[:,neig]

    # 步骤五 计算主成分分析后的图像 np.dot()
    pca = np.dot(newData, nvec)

    #步骤六 替换第一分量
    pca[:,0] = pan.ravel()

    #步骤七 逆变换
    inverse = np.dot(pca, nvec.T)
    meanres = mean[neig]
    inverse = inverse + meanres

    temp = np.zeros(mul.shape, dtype=np.float64)
    for i in range(3):
        temp[:, :, i] = (inverse[:, i]).reshape(row, col)

    # 转换到8位整型数据
    for i in range(dim):
        imax = np.max(temp[:, :, i])
        imin = np.min(temp[:, :, i])
        dst[:, :, i] = np.uint8(255.0 / (imax - imin + 0.0000001) * (temp[:, :, i] - imin))

    #直方图均衡化
    for i in range(dim):
        dst[:, :, i] = cv2.equalizeHist(dst[:, :, i])

    return dst


mul = cv2.imread('mul_input.tif', cv2.IMREAD_COLOR)
pan = cv2.imread('pan_input.tif', cv2.IMREAD_GRAYSCALE)

dst = myPCA(mul, pan)
plt.hist(dst[:,:,0].ravel(),64,(0,256))
plt.show()

cv2.imshow('pca_merge', dst)
cv2.waitKey(0)

