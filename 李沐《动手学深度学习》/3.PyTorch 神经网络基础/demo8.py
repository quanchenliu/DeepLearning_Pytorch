# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 参数绑定与参数共享

import torch
from torch import nn

X = torch.rand(2, 4)
shared = nn.Linear(8, 8)            # 我们只需要给共享层提供一个名称，以便可以引用它的参数
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
net(X)

# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])

# 以确保它们实际上是同一个对象，而不是仅有相同的值
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])


