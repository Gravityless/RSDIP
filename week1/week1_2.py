#coding=UTF-8

import numpy as np

class Matrix_Error(ValueError):
    pass

def mutiply_check(a,b):
    if a.shape[1]!=b.shape[0]:
        raise Matrix_Error('不符合矩阵相乘条件！')

def matrix_multiply(a,b):
    mutiply_check(a,b)

    c = np.zeros([a.shape[0],b.shape[1]], dtype = np.float32)

    for row in range(a.shape[0]):
        for col in range(b.shape[1]):
            d = np.multiply(a[row],b[...,col])
            for num in range(d.shape[0]):
                c[row][col] += d[num]
    return c

def main():
    a = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype = np.float32)
    b = np.array([[5,8,3],[9,4,2],[1,6,7]], dtype = np.float32)

    result = matrix_multiply(a,b)

    print(result)
    print(np.dot(a, b))

if __name__=='__main__':
    main()