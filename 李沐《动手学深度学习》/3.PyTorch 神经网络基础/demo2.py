# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 自定义一个块

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()                  # 调用父类的构造函数完成初始化
        self.hidden = nn.Linear(20, 256)    # 隐藏层
        self.out = nn.Linear(256, 10)       # 输出层

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

def define_self():
    net = MLP()
    X = torch.rand(2, 20)
    print(net(X).shape)

if __name__ == "__main__":
    define_self()