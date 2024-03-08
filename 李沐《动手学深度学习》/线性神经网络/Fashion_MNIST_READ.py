# -*- coding: utf-8 -*-
# @Time    : 2024/3/7
# @Author  : quanchenliu
# @Function: 如何读取多类分类的数据集，本例使用Fashion-MNIST数据集

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()               # 设置图形显示格式为SVG

"""定义一个函数，返回用于加载数据的工作进程数量"""
def get_dataloader_worker():    # @save
    return 4                    # 采用4个进程来读取数据


"""用于将 Fashion-MNIST 数据集的数字标签索引转换为相应的文本标签"""
def get_fashion_mnist_labels(labels):   # @save
    text_lables = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_lables[int(i)] for i in labels]


"""创建函数来可视化这些样本"""
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): # @save
    figsize = (num_cols * scale, num_rows * scale)                      # 计算网格大小
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)     # 创建一个包含指定行数和列数的网格图形，并返回图形对象和轴对象的元组
    axes = axes.flatten()                                               # 将二维数组的轴对象展开成一维数组

    # 使用enumerate函数遍历图像和对应的轴对象，zip函数将图像和轴对象配对。
    # i是索引，ax是轴对象，img是图像。
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):                                        # 检查当前图像是否是 PyTorch 张量
            ax.imshow(img.numpy())                                      # 将其转换为 NumPy 数组，并使用 imshow 方法显示在当前轴对象上
        else:
            ax.imshow(img)                                              # 否则，直接使用ax.imshow方法显示在当前轴对象上

        # 将轴对象的x轴、y轴标签设为不可见，以去除图像的轴刻度。
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    return axes


"""定义一个函数，用于加载Fashion-MNIST数据集的训练集和测试集"""
def load_data_fashion_mnist(batch_size, resize=None):       # @save
    trans = [transforms.ToTensor()]                         # 通过 ToTensor 实例将图像数据从 PIL 类型转变为 32 位浮点数格式
    if resize:                                              # 如果指定了resize，则调整图像大小
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)                       # 创建一个数据转换管道，将上述定义的所有转换操作组合成一个整体的数据转换过程。

    # root参数指定数据集的根目录，train参数表示加载训练集，download参数表示是否需要下载数据集。
    # transform参数指定数据转换操作（在本例中是将图像转换为张量）
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

    # 一个包含训练集和测试集数据加载器的元组, 训练集打乱顺序, 测试集不打乱顺序
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_worker()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_worker()))


def test_function():
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break

if __name__ == "__main__":
    test_function()