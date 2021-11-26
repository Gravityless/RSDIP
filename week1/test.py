#coding=UTF-8

class Matrix_Error(ValueError):
    pass

import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype = np.float32)
b = np.array([[5,8,3],[9,4,2],[1,6,7]], dtype = np.float32)

c = np.zeros([a.shape[0],b.shape[1]], dtype = np.float32)

import matplotlib.pyplot as plt
import numpy as np
x = [1,2,3,4,5,6]
y = [1,2,3,4,5,6]
plt.plot(x,y)
plt.show()

# try:
#     10 / 0
# except ZeroDivisionError:
# raise Matrix_Error('矩阵错误!')
# print('error')

for row in a:
    print(row)
for col in range(b.shape[1]):
    print(b[...,col])
