# -*- coding: utf-8 -*-
# @Time    : 2024/3/5
# @Author  : quanchenliu
# @Function: Numpy 的定义与数据运算

import random
import numpy as np

# 1、创建 Numpy 数组
a = np.array([1, 2, 3, 4, 5])
b = np.array(range(1, 6))
c = np.arange(1, 6)
print(a, b, c)                              # a b c 的内容是一致的，注意区别 arange 和 range

# 2、Numpy 数组的数据类型操作
t1 = np.array(range(1, 6), dtype=int)       # 强制指定 t1 中的数据类型
t2 = t1.astype("bool")                      # 修改数据类型
print(t1, t1.dtype, t2, t2.dtype)

# 3、Numpy 数组取固定位数的小数
t3 = np.array([random.random() for i in range(10)])         # 调用 random.random()，生成[0,1)之间的随机数
t4 = np.round(t3, 2)
print(t4)

# 4、Numpy 数组的形状
t5 = np.array(range(24))
t6 = t5.reshape(-1, 4)                      # [6, 4]
t7 = t5.reshape(2, 3, 4)                    # 2 个 3行4列的数组
print('t5.shape=', t5.shape, ', t6.shape=', t6.shape, ', t7.shape=', t7.shape)
t8 = t7.flatten()                           # 将一个多维数组展开成一维数组
print(t8, t8.shape)

# 5、Numpy 数组的计算——按元素计算、广播机制
t9 = t8 + 2
print(t9)

# 6、降维求和
t10 = t6.sum()                              # 调用 sum() 求和，结果是一个标量
t11 = t6.sum(axis=0)                        # 指定沿轴 0 降维求和，即：列和
t12 = t6.sum(axis=1)                        # 指定沿轴 1 降维求和，即：行和
print(t12)

print(t7)                                   # [2,3,4]
t13 = t7.sum()                              # 调用 sum() 求和，结果是一个标量
t14 = t7.sum(axis=0)                        # 2个[3,4]的矩阵按元素相加得到一个[3，4]的矩阵
t15 = t7.sum(axis=1)                        # 2个[3,4]的矩阵各自求列和得到2个[1,4]的矩阵，然后拼接得到一个[2,4]的矩阵
t16 = t7.sum(axis=2)                        # 2个[3,4]的矩阵各自求行和得到2个[1,3]的矩阵，然后拼接得到一个[2,3]的矩阵
print(t16)