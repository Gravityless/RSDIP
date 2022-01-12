import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('homework_img.tif')
Z = img.reshape((-1,3))
rows, cols, dims = img.shape
res2 = np.zeros((img.shape), dtype=np.float32)

# convert to np.float32
Z = np.float32(Z)

# 定义颜色
cmap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 57, 77], [0, 255, 255]])

# 分别输出迭代1~9次的结果
for i in range(9):
    # define criteria and K
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, i+1, 1.0)
    K = 5
    ret,label,center = cv.kmeans(Z,K,None,criteria,0,cv.KMEANS_PP_CENTERS)

    # match color map
    res = cmap[label.flatten()]
    for n in range(3):
        res2[:,:,n] = res[:,n].reshape(rows, cols)

    plt.subplot(3, 3, (i + 1))
    plt.imshow(res2.copy()[:, :, ::-1])  ##BGR到RGB
    plt.title(str(i + 1))
    plt.xticks([])
    plt.yticks([])

plt.show()