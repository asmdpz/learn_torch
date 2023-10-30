import torch
from torch import nn


# 6.3 填充和步幅
# 6.3.1 填充
# 定义一个计算卷积的函数
def comp_conv2d(conv2d, X):
    # 这里的(1, 1)表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道数
    return Y.reshape(Y.shape[2:])


# 每个侧边都填充了1行或1列，共填充了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
Y = comp_conv2d(conv2d, X)
print(Y.shape)


# 6.3.2 步幅
# torch.Size([4, 4])
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
Y = comp_conv2d(conv2d, X)
print(Y.shape)
# torch.Size([2, 2])
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

