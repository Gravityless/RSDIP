import numpy as np

# 创建了三维的 ndarray
a = np.arange(8).reshape(2, 2, 2)

print('原数组：')
print(a)
print('获取数组中一个值：')
print(np.where(a == 6))
print(a[1, 1, 0])  # 为 6
print('\n')

# 将轴 2 滚动到轴 0（宽度到深度）

print('调用 rollaxis 函数：')
b = np.rollaxis(a, 2, 0)
print(b)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [0, 1, 1]
# 最后一个 0 移动到最前面
print(np.where(b == 6))
print('\n')

# 将轴 2 滚动到轴 1：（宽度到高度）

print('调用 rollaxis 函数：')
c = np.rollaxis(a, 2, 1)
print(c)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [1, 0, 1]
# 最后的 0 和 它前面的 1 对换位置
print(np.where(c == 6))
print('\n')