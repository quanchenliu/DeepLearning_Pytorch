# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 自定义不带参数的层

import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):           # 减去均值
        return X - X.mean()

layer = CenteredLayer()
X = torch.FloatTensor([1, 2, 3, 4, 5])
y = layer(X)
print(y)