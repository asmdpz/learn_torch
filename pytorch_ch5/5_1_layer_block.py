import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 5.1 层和块
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256,10)
)
X = torch.rand(2, 20)
Y = net(X)
print(Y)


# 5.1.1 自定义块
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


net = MLP()
Y = net(X)
print(Y)


# 5.1.2 顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            # _modules中，_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
Y = net(X)
print(Y)


# 5.1.3 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层，这相当于两个全连接层共享参数
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
Y = net(X)
print(Y)


class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(
    NestMLP(),
    nn.Linear(16, 20),
    FixedHiddenMLP()
)
Y = chimera(X)
print(Y)