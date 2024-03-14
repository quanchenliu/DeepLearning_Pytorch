# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 相比于直接使用 Sequential 类，自定义块具有更高的灵活性

import torch
from torch import nn
from torch.nn import functional as F

class FixHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight))
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

def main():
    X = torch.rand(2, 20)
    net = FixHiddenMLP()
    y = net(X)
    print(y)

if __name__ == "__main__":
    main()