# -*- coding: utf-8 -*-
# @Time    : 2024/3/19
# @Author  : quanchenliu
# @Function: LeNet(sigmoid 激活函数+平均池化)
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

"""评估精度"""
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    if isinstance(net, nn.Module):
        net.eval()                                              # 设为评估模式
        if not device:
            # net.parameters(): 返回模型中所有可学习参数，每个参数都表示为张量
            # iter(): 将模型参数迭代器转换为一个可迭代对象
            # next(): 用于从迭代器中获取下一个元素（这里是第一个参数）
            # .device: 用于获取张量所在的设备
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(2)                     # 创建一个累加器，跟踪两个值，正确预测数、样本总数
    with torch.no_grad():                           # 不计算梯度，从而减少内存占用和提高速度
        for X, y in data_iter:
            if isinstance(X, list):                 # 检查输入数据 X 是否为列表
                X = [x.to(device) for x in X]       # 使用 to(device) 方法将张量 x 移动到指定的设备上，并将移动后的张量保存回列表 X 中
            else:
                X = X.to(device)
            y = y.to(device)                        # 将标签移动到指定的设备上

            nums = d2l.accuracy(net(X), y)          # 计算正确预测数
            metric.add(nums, y.numel())

    return metric[0]/metric[1]



"""训练函数"""
#@save
def train_gpu(net, train_iter, test_iter, num_epochs, lr, device):

    # 权重初始化函数, 对卷积层和全连接层进行 Xavier 初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)                                     # 调用 init_weights 函数，并将其应用到模型 net 的所有层上
    net.to(device)                                              # 将模型移动到指定的设备上
    print('training on', device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)        # 定义优化器, 采用随机梯度下降法进行优化
    loss = nn.CrossEntropyLoss()                                # 交叉熵损失函数（CrossEntropyLoss）
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)                 # 累加器, 记录训练损失、正确数量、样本总量
        net.train()                                 # 设为训练模式

        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()                   # 清空梯度，避免梯度累积
            X, y = X.to(device), y.to(device)
            y_hat = net(X)                          # 计算预测值
            l = loss(y_hat, y)                      # 计算损失
            l.backward()                            # 计算梯度
            optimizer.step()                        # 更新网络参数

            with torch.no_grad():
                total_loss = l*X.shape[0]           # 损失 * 批量大小 = 该批量的总损失
                acc = d2l.accuracy(y_hat, y)        # 计算预测正确的数量
                metric.add(total_loss, acc, X.shape[0])

            train_l = metric[0]/metric[2]           # 计算训练损失
            train_acc = metric[1]/metric[2]         # 计算训练的正确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (train_l, train_acc, test_acc))
        if epoch == num_epochs - 1:
            plt.show()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')



def main():
    batch_size = 256
    lr = 0.5                # 如果使用sigmoid激活、平均池化，设置学习率为0.9
    num_epochs = 10
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    # 用Flatten函数将矩阵数据拉直
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    # d2l.try_gpu(), 用于获取可用的 GPU 设备
    train_gpu(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

if __name__ == "__main__":
    main()