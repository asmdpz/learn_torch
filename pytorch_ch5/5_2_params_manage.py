import torch
from torch import nn

# 5.2 参数管理
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
X = torch.rand(size=(2, 4))
Y = net(X)
print(Y)

# 5.2.1 参数访问
print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias.data)
print(net[2].bias)
print(net[2].weight.grad is None)

print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

data = net.state_dict()['2.bias'].data
print(data)


def block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


rgnet = nn.Sequential(
    block2(),
    nn.Linear(4, 1)
)
Y = rgnet(X)
print(Y)
print(rgnet)


# 5.2.2 参数初始化
def init_norm(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std = 0.01)
        nn.init.zeros_(m.bias)


net.apply(init_norm)
w0 = net[0].weight.data[0]
b0 = net[0].bias.data[0]
print(w0)
print(b0)


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        '''
        进行判定，看每一个权重的绝对值是否大于等于5，
        如果大于等于5则证明在(5, 10)和(-10，-5)区间上，那返回true，也就是1，
        m.weight.data乘1数值不变；反之会返回false，也就是0
        '''
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:2])
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

# 5.2.3 参数绑定
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 1)
)
Y = net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
