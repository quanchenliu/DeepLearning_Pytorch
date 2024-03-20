# -*- coding: utf-8 -*-
# @Time    : 2024/3/20
# @Author  : quanchenliu
# @Function: ResNet

import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

class Residual(nn.Module):  #save
    def __init__(self, in_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)    # 卷积核大小为1，做恒等运算
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))     # [B, C, H, W] 或 [B, C, H/2, W/2]
        y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)                        # [B, C, H, W] 或 [B, C, H/2, W/2]

def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    block_list = []
    for i in range(num_residuals):
        if i == 0 and not first_block:                                      # i=0且不是第一个模块
            block_list.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
        else:                                                               # 是第一个模块
            block_list.append(Residual(num_channels, num_channels))         # 第一个块要求输入通道数与输出通道数相同
    return block_list

def main():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),        # [B, 64, H/2, W/2]
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))            # [B, 64, H/4, W/4]
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))                  # [B, 64, H/4, W/4]
    b3 = nn.Sequential(*resnet_block(64, 128, 2))                                   # [B, 128, H/8, W/8]
    b4 = nn.Sequential(*resnet_block(128, 256, 2))                                  # [B, 256, H/16, W/16]
    b5 = nn.Sequential(*resnet_block(256, 512, 2))                                  # [B, 512, H/32, W/32]

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),                               # [B, 512, 1, 1]
                        nn.Flatten(),                                               # [1, 512*1*1]
                        nn.Linear(512, 10))                                         # [1, 10]

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()

if __name__ == "__main__":
    main()