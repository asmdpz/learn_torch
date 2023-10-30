import torch
from torch import nn
from d2l import torch as d2l


# 6.5 汇聚层（池化层）
# 6.5.1 最大池化和平均池化
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_h].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_h].mean()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])
Y = pool2d(X, (2, 2), 'avg')
print(Y)

# 6.5.2 填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
max_pool2d = nn.MaxPool2d(3)
Y = max_pool2d(X)
print(Y)
max_pool2d = nn.MaxPool2d(3, padding=1, stride=2)
Y = max_pool2d(X)
print(Y)
max_pool2d = nn.MaxPool2d((2, 3), padding=(0, 1), stride=(2, 3))
Y = max_pool2d(X)
print(Y)

# 6.5.3 多个通道
# 池化层的输入输出通道数相同
X = torch.cat((X, X+1), 1)
print(X)
max_pool2d = nn.MaxPool2d(3, padding=1, stride=2)
Y = max_pool2d(X)
print(Y)