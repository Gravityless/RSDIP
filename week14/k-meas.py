import cv2
import numpy as np
import matplotlib.pyplot as plt


##计算两个数列之间的距离
def distance(p1, p2):
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist


##类中心初始化
def initCentroids(dataset, k):
    # 2.3.1 定义相关变量
    numSample, dims = dataset.shape
    centroids = np.zeros((k, dims), np.float32)  # 类中心值

    # 2.3.2 计算类中心的位置
    minCol = np.min(dataset, axis=0)  # 计算各波段中最小值
    maxCol = np.max(dataset, axis=0)  # 计算各波段中最大值
    rangeCol = maxCol - minCol  # 各波段中像元值范围
    step = rangeCol / k  # 分成k类后，每类的步长

    # 2.3.3 类中心初始化
    for i in range(k):
        centroids[i, :] = minCol + (i + 1) * step
    return centroids


##构建K-means分类器
def kmeans(dataset, k, maxIterationNum):
    # 2.1 定义相关变量
    resultList = []  # 定义变量存储结果
    numSample, dim = dataset.shape
    # 存放每一像元的归属类别标签和到该类的距离
    clusterAssment = np.zeros((numSample, 2), dtype=np.float32)
    clusterFlag = True  # 类别标记，判断迭代是否继续
    numIteration = 0  # 迭代次数

    # 2.2 设定初始距离
    for i in range(numSample):
        clusterAssment[i, 1] = 10000
    # 2.3 生成初始聚类中心
    centroid = initCentroids(dataset, k)

    # 2.4 聚类
    while (clusterFlag):
        # 2.4.1 复制上一次聚类结果
        clusterAssmentBefore = clusterAssment.copy()

        # 2.4.2 迭代次数判断
        numIteration += 1
        if (numIteration > maxIterationNum):
            break  # 如果迭代次数超过指定次数，跳出循环
        print('第%s次迭代中...' % numIteration)

        # 2.4.3 更改循环判别条件为不循环，如果下文条件允许循环则再次更改为True
        clusterFlag = False

        # 2.4.4 循环聚类
        for i in range(numSample):  # 遍历每个像元
            # 2.4.4.1 设置初始距离和标签值
            minDist = 10000000
            minIndex = 0  # 类别标签初始值

            # 2.4.4.2 距离计算
            for j in range(k):
                # 逐一计算每一像元与各类中心的距离
                dis = distance(dataset[i, :], centroid[j, :])
                if (dis < minDist):
                    # 如果与某一类别中心的距离较小，则判定像元属于该类别
                    minDist = dis
                    minIndex = j

            # 2.4.4.3 判别
            if (clusterAssment[i, 0] != minIndex):  # 若类别发生改变则执行下次迭代
                clusterFlag = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        # 2.4.5 判断是否收敛
        distAfter = np.sum(clusterAssment[:, 1])  # 本次迭代结果(距离)
        distBefore = np.sum(clusterAssmentBefore[:, 1])  # 上次迭代结果(距离)
        if (distAfter > distBefore):  # 结果发散，舍弃本次
            return resultList

        # 2.4.6 重新计算类别中心
        labelMat = clusterAssment[:, 0]  # 取出第一列(类别标签)
        for cent in range(k):  # 针对每个类别重新计算
            temp = dataset[labelMat == cent]  # 分别选取属于每一类的像元
            centroid[cent, :] = np.mean(temp, axis=0)  # 计算本类别新的中心

        # 2.4.7 将本次计算结果添加到结果列表中
        resultList.append((centroid.copy(), clusterAssment.copy()))
    return resultList


##第一步：读入图像，并进行分类前的处理
# 1.1 读入图像
fileName = 'homework_img.tif'
img = cv2.imread(fileName, cv2.IMREAD_COLOR)

# 1.2 将图像转换成数组形式
rows, cols, dims = img.shape
dataset = np.zeros((rows * cols, dims), dtype=np.float32)
for i in range(dims):
    dataset[:, i] = img[:, :, i].ravel()

# 1.3 设置分类参数
num = 5  # 类别数
maxIteration = 9  # 最大迭代次数

##第二步：调用函数，进行K-means分类
# 2.0 调用函数，执行分类
resultList = kmeans(dataset, num, maxIteration)

##第三步：显示结果
res = np.zeros(img.shape, dtype=np.float32)
cmap = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 57, 77], [0, 255, 255]]  # 定义颜色

for i in range(len(resultList)):  # 取出每次迭代的结果
    centroid, clusterAssment = resultList[i]
    labelMat = clusterAssment[:, 0]  # 取出类别标签
    for m in range(num):
        dataset[labelMat == m] = cmap[m]  # 每个类别赋颜色值
    for n in range(3):
        res[:, :, n] = (dataset[:, n]).reshape((rows, cols))
    plt.subplot(3, 3, (i + 1))
    plt.imshow(res.copy()[:, :, ::-1])  ##BGR到RGB
    plt.title(str(i + 1))
    plt.xticks([])
    plt.yticks([])
plt.show()