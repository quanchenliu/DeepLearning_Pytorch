# -*- coding: utf-8 -*-
# @Time    : 2024/3/9
# @Author  : quanchenliu
# @Function: 从零实现多层感知机

import torch
from d2l import torch as d2l
from torch import nn
import matplotlib.pyplot as plt

# 定义激活函数
def ReLU(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 定义模型
def net(X):
    X = X.reshape(-1, num_inputs)         # [28, 28] --> [1, 784]
    H = ReLU(X @ W1 + b1)                 # @ 表示矩阵乘法
    return (H @ W2 + b2)

def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 初始化模型参数
    global num_inputs, num_outputs, num_hiddens, W1, b1, W2, b2, params
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)      # 为什么要随机
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')                # 交叉熵损失函数

    # 训练
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr)                       # 选择使用随机梯度下降的方法进行训练
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    # 评估
    d2l.predict_ch3(net, test_iter)

    # 显示图像
    plt.show()
if __name__ == "__main__":
    main()