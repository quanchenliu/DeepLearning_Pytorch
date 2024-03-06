# -*- coding: utf-8 -*-
# @Time    : 2024/3/5
# @Author  : quanchenliu
# @Function: Numpy 读取 CSV 文件

import numpy as np
import csv

road = r'C:\Users\DELL\Desktop\深度学习小白进阶之路\李沐《动手学深度学习》\一些重要的第三方库的学习\NumpyLearning\data.csv'
# 1、简单读取: 默认情况下，数据被认为是float类型，因此以使用str参数，让方法读取数据时，支持str类
data_all = np.loadtxt(road, str, delimiter=',')         # 指定了CSV文件中字段之间的分隔符，默认情况下为逗号
print(data_all)                                         # 打印加载的数据
print('////////////////////////////////////////////////////////////')

# 2、不读标题/跳过首行:skiprow=1
data = np.loadtxt(road, str, delimiter=",", skiprows=1)
print(data)
print('////////////////////////////////////////////////////////////')

# 3、访问数据的形状、维度和大小
print("Shape:", data.shape)                             # 打印数组的形状
print("Dimensions:", data.ndim)                         # 打印数组的维度
print("Size:", data.size)                               # 打印数组的大小
print('////////////////////////////////////////////////////////////')

# 4、访问数组的指定行/列
print("First row:", '\n', data[0])                      # 打印数组的第一行
print("Fifth column:", '\n', data[:, 4])                # 打印数组的第五列
print("1~5 rows:", '\n', data[:5, :])                   # 打印前5行
print('////////////////////////////////////////////////////////////')

# 5、对数组进行统计计算: 此时由于CSV文件中存在数据类型混合的情况，因此需要对数据进行清洗和转换
numeric_data = data[:, 4:].astype(float)                # 将第1~4列的数据转换为浮点型数据
print("Mean value:", np.mean(numeric_data))             # 计算数据的均值
print("Standard deviation:", np.std(numeric_data))      # 计算数组的标准差
print('////////////////////////////////////////////////////////////')

# 6、只读取特定列: usecols参数
data_special_list = np.loadtxt(road, str, delimiter=",", skiprows=1, usecols=(0, 1, 2))
print(data_special_list)                                # 在usecols参数中确定读取哪几行，及这几行的读取顺序
print('////////////////////////////////////////////////////////////')

# 7、numpy切片: 从Numpy数组中选择特定的元素子集的操作















