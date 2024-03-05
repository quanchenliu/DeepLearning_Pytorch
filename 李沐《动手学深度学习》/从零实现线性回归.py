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

def draw(feature, labels):
    d2l.set_figsize()
    d2l.plt.scatter(feature[:, 1].detach().numpy(), labels.detach().numpy(), 1)

def main():
    true_w = torch.tensor([2, -3.4])  # [2]
    true_b = 4.2
    feature, labels = synthetic_data(true_w, true_b, 1000)  # [1000, 2], [1000, 1]
    draw(feature, labels)

if __name__ == "__main__":
    main()
