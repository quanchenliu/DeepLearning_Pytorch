# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 构建一个简化的 Sequential 块(把其他模块串起来)

import torch
from torch import nn

class MySequential(nn.Module):
    def __init__(self, *args):              # args: list of argument, 即: 需要按序执行的各个模块/层
        super().__init__()
        for idx, block in enumerate(args):
            self._modules[str(idx)] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

def main():
    X = torch.rand(2, 20)
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    print(net(X).shape)

if __name__ == "__main__":
    main()