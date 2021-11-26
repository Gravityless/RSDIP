import numpy as np
import matplotlib.pyplot as plt

# 定义原始数据
ary = np.array([183, 183, 175, 159, 159, 160, 151, 155, 166, 159, 166, 164, 159, 164, 167, 163, 156, 163, 157, 155], dtype = np.float64)
template = np.array([1, 1, 1], np.float64) / 3.0     # 定义模板
ans = ary.copy()                             # 定义数组，存储运算结果

# 卷积运算
for i in range(1, len(ary)-1):
    temp = ary[i-1: i+2]
    ans[i] = np.sum(temp * template)

# 绘制原始数据与结果
plt.plot(range(20), ary, linewidth = 2, color = 'b')
plt.plot(range(20), ans, linewidth = 2, color = 'r')
plt.legend(('before', 'after'), loc = 'best')
plt.show()
