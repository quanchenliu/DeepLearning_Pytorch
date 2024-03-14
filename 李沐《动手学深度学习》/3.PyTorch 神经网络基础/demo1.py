# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : quanchenliu
# @Function: 简洁实现一个单隐藏层的MLP

import torch
from torch import nn

def use_model():
    # 批量大小: 2, 输入特征维度: 20, 生成 [0, 1) 均匀分布的随机数
    X = torch.rand(2, 20)
    # 隐藏的隐藏单元数: 256, 输出特征维度大小: 10
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    print(net(X).shape)

if __name__ == "__main__":
    use_model()