# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 自定义初始化

from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()])
        nn.init.uniform_(m.weight, -10, 10)

        # m.weight.data.abs() >= 5 会生成一个相同形状的布尔类型的矩阵，其中:
        #   True: 对应位置的权重的绝对值 >= 5
        #   False表示小于5
        m.weight.data *= m.weight.data.abs() >=5

net.apply(my_init)
print(net[0].weight)

# 我们始终可以直接设置参数
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])