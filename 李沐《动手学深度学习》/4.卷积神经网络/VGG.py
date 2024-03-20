# -*- coding: utf-8 -*-
# @Time    : 2024/3/20
# @Author  : quanchenliu
# @Function: VGG

import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

"""实现 VGG 块: num_convs 是卷积层数量, in_channels 是输入通道数, out_channels 是输出通道数"""
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))       # [H, W]
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))                                    # [H/2, W/2]
    return nn.Sequential(*layers)

"""定义 VGG-11 网络"""
def vgg_11(conv_arch):
    conv_blocks = []
    in_channels = 1

    # 卷积层与池化层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    # 全连接层
    return nn.Sequential(*conv_blocks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                         nn.Linear(4096, 10))

def main():
    lr, num_epochs, batch_size, ratio = 0.05, 10, 123, 4
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    small_conv_arch = [(pair[0], pair[1]//ratio) for pair in conv_arch]
    net = vgg_11(small_conv_arch)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()


if __name__ == "__main__":
    main()