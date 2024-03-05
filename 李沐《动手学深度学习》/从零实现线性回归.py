# -*- coding: utf-8 -*-
# @Time    : 2024/3/5
# @Author  : quanchenliu
# @Function: 从零开始实现线性回归模型

import random
import torch
from d2l import torch as d2l

"""生成数据集"""
def synthetic_data(w, b, num_examples):  # @save
    ''' torch.matmul() 函数几乎可以用于所有矩阵或向量相乘的情况，其乘法规则视参与乘法的两个张量的维度而定: y = torch.matmul(X, w) + b
        num_examples 指定了生成的样本数量, len(w)指定了特征的数量, 即权重向量w的长度'''

    X = torch.normal(0, 1, (num_examples, len(w)))  # X ~ N(0,1),[1000, 2]
    y = torch.mv(X, w) + b                          # 通过矩阵乘法计算标签 y, [1000]
    y += torch.normal(0, 0.01, y.shape)             # 按元素加法, 向标签 y 中添加一些噪声以模拟真实世界中的数据
    return X, y.reshape((-1, 1))                    # [1000, 2], [1000, 1]

"""读取数据集"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)                    # 返回张量的第一个维度的长度, 1000
    indices = list(range(num_examples))             # 创建一个包含所有样本索引的列表

    # 这个函数在深度学习中常用于数据集的随机打乱，以增加训练的随机性，防止模型过拟合。
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        # 使用 yield 关键字生成当前批量的特征和标签，并在下次迭代中继续执行
        yield features[batch_indices], labels[batch_indices]        # [10, 2], [10,1]

"""定义模型"""
def linreg(X, w, b):    # @save
    return torch.matmul(X, w) + b

"""定义损失函数"""
def squared_loss(y_hat, y):     # @save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2        # 预测值 y_hat, 真实值 y

"""定义优化算法——小批量随机梯度下降"""
def sgd(params, lr, batch_size):    # @save
    with torch.no_grad():                                   # 以下操作无需计算梯度，不被记录进计算图，不被用于自动求导
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()                              # 将参数 param 的梯度张量清零

def main():
    true_w = torch.tensor([2, -3.4])                            # [2]
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)     # [1000, 2], [1000, 1]

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 初始化权重 w.shape = [2,1]
    b = torch.zeros(1, requires_grad=True)                      # 初始化偏置值为0
    lr = 0.03                                                   # 设置学习率为 0.03
    num_epochs = 3                                              # 设置轮数为 3
    batch_size = 10                                             # 批处理大小

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = squared_loss(linreg(X, w, b), y)                # X 和 y 的小批量损失
            l.sum().backward()                                  # l.shape = [batch_size, 1], 而非标量，因此需降维求和成标量后计算梯度
            sgd([w, b], lr, batch_size)                         # 使用优化算法更新模型参数 w, b

        with torch.no_grad():                                   # 计算每个迭代周期的误差
            train_l = squared_loss(linreg(features, w, b), labels)
            print(f'epoch{epoch+1}, loss{float(train_l.mean()):f}')

    print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差:{true_b - b}')

if __name__ == "__main__":
    main()
