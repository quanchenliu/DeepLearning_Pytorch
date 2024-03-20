# -*- coding: utf-8 -*-
# @Time    : 2024/3/19
# @Author  : quanchenliu
# @Function: AlexNet

import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

def main():

    # X = [B, C, H, W] = [1, 1, 224, 224]
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),       # [1, 96, 54, 54]
        nn.MaxPool2d(kernel_size=3, stride=2),                                  # [1, 96, 26, 26]

        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),                # [1, 256, 26, 26], 填充为2, 即填充之后的形状为:[1, 96, 30, 30]
        nn.MaxPool2d(kernel_size=3, stride=2),                                  # [1, 256, 12, 12]

        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),               # [1, 384, 12, 12]
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),                                  # [1, 256, 5, 5]

        nn.Flatten(),                                                           # [1, 256*5*5 = 6400]
        nn.Linear(6400, 4096), nn.ReLU(),                                       # [1, 4096]
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10))                                                    # [1, 10]

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    lr = 0.01
    num_epochs = 10
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()

if __name__ == "__main__":
    main()