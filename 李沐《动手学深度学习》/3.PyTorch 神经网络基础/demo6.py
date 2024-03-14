# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 内置初始化

from torch import nn

# 1、将权重初始化为 N(0, 0.0001), 偏置初始化为0
def init_normal(m):                 # m 是一个 Module
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net.apply(init_normal)
print(net[0].weight.data)
print(net[0].bias.data)

# 2、对不同层应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net1[0].apply(init_xavier)
net1[2].apply(init_constant)
print(net1[0].weight.data)
print(net1[2].weight.data)