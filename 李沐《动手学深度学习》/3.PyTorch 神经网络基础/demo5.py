# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 访问参数

import torch
from torch import nn

# 1、构造一个MLP
X = torch.rand(2, 4)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net(X)

# 2、通过索引访问模型的任意层, state_dict()方法可以返回该层的参数字典
print(net[2].state_dict())

# 3、访问某一层的某一个参数
print(net[2].bias, '\n', net[2].weight)

# 4、一次性访问所有参数, named_parameters() 方法返回一个生成器，生成器会依次产生参数的名称和参数值。
print(*[(name, param.shape) for name, param in net[2].named_parameters()])      # 访问顶层
print(*[(name, param.shape) for name, param in net.named_parameters()])         # 访问所有层

# 5、如果一个块是嵌套块, 如何访问参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # f'block{i}': 用于生成子模块的名称
        # block1()   : 要添加的子模块
        net.add_module(f'block{i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)
print(rgnet[0][1][0].bias.data)                     # 访问 第一个块的 第二个子块的 第一层 的偏置