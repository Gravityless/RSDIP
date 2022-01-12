import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

## 1. 以表格式读取训练样本数据
data_table = pd.read_csv('sample_training.csv')
#将数据转到numpy库可读写的格式
data = np.array(data_table)

X = data[:, 2:5]    # X变量指用于分类的特征
Y = data[:, 5]      # Y变量指类别标签
#print(X)
#print(Y)

## 2. 利用训练样本训练模型
clf = RandomForestClassifier(n_estimators = 1000, max_features = None)
clf.fit(X, Y)

image = cv2.imread('NJU.tif', cv2.IMREAD_COLOR)
rows, cols, dims = image.shape  
image = image.reshape(rows * cols, dims)


## 3. 用构建好的模型对整幅图像进行分类
result = clf.predict(image).reshape(rows, cols)
cv2.imwrite('result.tif', result)

## 4. 精度评价
#利用验证样本评估分类结果
data_validation = pd.read_csv('sample_validation.csv')
data_validation = np.array(data_validation)
ft = data_validation[:, 2:5]    # 用于分类的特征
label0 = data_validation[:, 5]      # 类别标签
label = label0.tolist()             # 转成list

pl0 = clf.predict(ft)   # 返回预测标签
pl = pl0.tolist()

# 计算并输出各精度指标
accuracy = accuracy_score(label, pl)
print('Overal accuracy: %.2f' % (accuracy * 100) + '%')
kappa = cohen_kappa_score(label, pl)
print('Kappa: %.2f'% kappa)
print('Confusion matrix:')
matrix = confusion_matrix(label, pl, labels=[0, 1, 2])
print(matrix)

# 输出图像
res2 = np.zeros((cols, rows, dims), dtype=np.uint8)
cmap = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
res = cmap[result.ravel()]
res = res.reshape(cols, rows, dims)
for n in range(3):
    res2[:, :, n] = res[:, :, n]
cv2.imshow('classified', res2)
cv2.waitKey(0)










