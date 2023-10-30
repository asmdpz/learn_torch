import torch
from torch import nn

# 5.6 GPU
# 5.6.1 计算设备
print(torch.cuda.device_count())


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())

# 5.6.2 张量与GPU
x = torch.tensor([1, 2, 3])
print(x.device)
X = torch.ones(2, 3, device=try_gpu())
print(X)

# 将 cuda:0 的张量 X 复制到 cuda:1 的张量 Z
# Z = X.cuda(1)
# 如果指定的gpu没有发生改变，就不会复制并分配新的内存
print(X.cuda(0) is X)

# 5.6.3 神经网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
Y = net(X)
print(Y)
print(net[0].weight.data.device)