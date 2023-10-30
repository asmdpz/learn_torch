import torch

# 2.1 数据操作
# 2.1.1 入门
# arange(12): 创建一个包含0到11的整数的行向量
x = torch.arange(12)
print(x)

# x.shape: shape属性访问张量x的形状
print(x.shape)

# x.numel(): 获取张量x的元素总数 12
print(x.numel())

# x.reshape(3, 4): 把张量x从形状(12,)的行向量转换为形状为(3,4)的矩阵
X = x.reshape(3, 4)
print(X)

# 全0
print(torch.zeros((2, 3, 4)))

# 全1
print(torch.ones((2, 3, 4)))

# 随机初始化参数，每个元素服从均值为0、标准差为1的正态分布
print(torch.randn(3, 4))

# 为张量的每个元素赋值
print(torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]))

# 2.1.2 运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y) # **是开方运算
print(torch.exp(x)) # e的x次方

X = torch.arange(12,dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 6x4 或 3x8
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

print(X == Y)

# 所有元素求和
print(X.sum())

# 2.1.3 广播机制
a = torch.arange(6).reshape((3, 2))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)

# 2.1.4 索引和切片
print(X[-1])
print(X[1:3])

# 第2行第3列的元素赋值为9
X[1, 2] = 9
print(X)

# 前2行所有元素赋值为12
X[0:2, :] = 12
print(X)

# 2.1.5 节省内存
# id(Y): 获取内存中引用对象Y的确切地址
before = id(Y)
Y = Y + X
print(id(Y) == before)

# 通过切片表示法将操作的结果分配给原地址
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)

# 2.1.6 转换为其他Python对象
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

# 张量转换为python标量
a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))
