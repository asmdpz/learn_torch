import torch
from d2l import torch as d2l


# 6.4 多输入多输出通道
# 6.4.1 多输入通道
def corr2d_multi_in(X, K):
    # 先遍历X和K的第0个维度（通道维度），再把它们加在一起
    # zip函数把每个通道的输入X与对应通道的卷积核K组队
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
multi_in = corr2d_multi_in(X, K)
print(X.shape)
print(K.shape)
print(multi_in)


# 6.4.2 多输出通道
def corr2d_multi_in_out(X, K):
    # 迭代K的第0个维度，每次都对输入X执行互相关运算
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(K)
multi_in_out = corr2d_multi_in_out(X, K)
print(multi_in_out)


# 6.4.3 1X1卷积层
# 使用全连接层实现1X1卷积
# 1X1卷积层需要的权重维度为 c_0 x c_i 加上一个偏置
def corr2d_multi_in_out_1X1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1X1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print("X", X)
print("K", K)
print("Y1", Y1)
print("Y2", Y2)
print("result:", float(torch.abs(Y1 - Y2).sum()) < 1e-5)
