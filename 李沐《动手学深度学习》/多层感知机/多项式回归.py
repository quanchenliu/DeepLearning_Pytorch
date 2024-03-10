# -*- coding: utf-8 -*-
# @Time    : 2024/3/10
# @Author  : quanchenliu
# @Function: 多项式回归

import math

import matplotlib.pyplot as plt
import torch
import numpy as np
from d2l import torch as d2l
from torch import nn


def evalutae_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        predict = net(X)
        y = y.reshape(predict.shape)
        l = loss(predict, y)
        metric.add(l.sum(), l.numel())

    return metric[0]/metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]

    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])

    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch+1) % 20 == 0:
            animator.add(epoch+1, (evalutae_loss(net, train_iter, loss), evalutae_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
    plt.show()


def main():
    max_degree = 20                                 # 多项式的最大阶数
    n_train, n_test = 100, 100                      # 训练集、测试集的大小
    true_w = np.zeros(max_degree)                   # 初始化权重, [20,]
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))     # 生成服从标准正态分布的随机数，形状为: [200, 1]
    np.random.shuffle(features)                                 # 将特征数据进行随机打乱

    # 生成多项式特征 poly_features, [200, 1] * [1, 20] = [200, 20]
    degree_array = np.arange(max_degree)                        # 创建一个从 0 到 19 的数组: [20,]
    degree_array = degree_array.reshape(1, -1)                  # [1, 20]
    poly_features = np.power(features, degree_array)            # np.power 用于实现数值的幂运算

    # 对生成的多项式特征进行归一化处理
    for i in range(max_degree):
        gamma = math.gamma(i + 1)                               # gamma(n) = (n-1)!
        poly_features[:, i] /= gamma                            # 一整列除以该列阶数的阶乘

    labels = np.matmul(poly_features, true_w)                   # labels.shape = [200,]
    noise = np.random.normal(scale=0.1, size=labels.shape)      # noise ~ N(0, 0.01)
    labels += noise

    # 转换参数为 tensor， 使其能够被 PyTorch 正确兼容
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32)
                                               for x in [true_w, features, poly_features, labels]]

    # 三阶多项式拟合
    # train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

    # 线性函数拟合——欠拟合
    # train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

    # 高阶多项式拟合——过拟合
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])

if __name__ == "__main__":
    main()