import cv2
import numpy as np
import matplotlib.pyplot as plt

def myPCA(mul, k):
    mul = np.float64(mul)

    # 步骤一 矩阵向量化 ravel()
    row, col, dim = mul.shape
    temp = np.zeros((row * col, dim), dtype = np.float32)
    normed = np.zeros((row * col, dim), dtype=np.float32)
    for i in range(dim):
        temp[:, i] = (mul[:, :, i]).ravel()

    # 步骤二 求图像数组的协方差矩阵 np.dot()
    for i in range(dim):
        normed[:, i] = temp[:, i] - np.mean(temp[:, i])

    scatter_matrix = np.dot(np.transpose(normed), normed)

    # 步骤三 计算特征值和特征向量 np.linalg.eig()
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)

    # 步骤四 选择特征向量组成变换矩阵 np.argsort()
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(dim)]
    eig_pairs.sort(reverse=True)

    # 步骤五 计算主成分分析后的图像 np.dot()
    feature = np.array(eig_pairs[k-1][1])
    data = np.dot(normed, np.transpose(feature))
    data = data.reshape(col, row)
    max = np.max(data)
    min = np.min(data)
    data = (data - min)/(max - min) * 255
    data = np.uint8(data)

    return data


mul = cv2.imread('mul_input.tif', cv2.IMREAD_COLOR)
img = myPCA(mul, 1)

# plt.hist(mul[2].ravel(), 50, [0, 256])
# plt.title('multiple first band')
# plt.show()
plt.hist(img.ravel(), 50, [0, 256])
plt.title('PCA first band')
plt.show()

cv2.imshow('this', img)
cv2.waitKey(0)

