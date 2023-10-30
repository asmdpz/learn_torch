
# 12.1 编译器和解释器

def add(a, b):
    return a + b


def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g


print(fancy_func(1, 2, 3, 4))


# 12.1.1 符号编程
def add_():
    return '''
def add(a, b):
    return a + b
'''


def fancy_func_():
    return """
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
"""


def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'


prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)

# 12.1.2 混合式编程

# 12.1.3 Sequential的混合式编程
import torch
from torch import nn
from d2l import torch as d2l


# 生产网络的工厂模式
def get_net():
    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    return net


x = torch.randn(size=(1, 512))
net = get_net()
Y = net(x)
print(Y)

# 通过使用 torch.jit.script 函数来转换模型
net = torch.jit.script(net)
Y = net(x)
print(Y)


#@save
class Benchmark:
    """
    用于度量运行时间
    """
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')


net = get_net()
with Benchmark('无torchscript'):
    for i in range(10000): net(x)

net = torch.jit.script(net)
with Benchmark('有torchscript'):
    for i in range(10000): net(x)

# 序列化
net.save('my_mlp')