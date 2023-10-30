import torch

# 2.3 线性代数
# 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y)
print(x * y)
print(x / y)
print(x ** y)

# 向量
x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

# 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

# 张量
X = torch.arange(24).reshape(2, 3, 4)
print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a + X).shape)

# 降维
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())
print(A.shape)
print(A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)
print(A.sum(axis=[0, 1])) # 结果与A.sum()相同
print(A.mean())
print(A.sum() / A.numel())
print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])
# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A)
# 按行累加计算
print(A.cumsum(axis=0))

# 点积
y = torch.ones(4, dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x, y))
print(torch.sum(x * y))

# 向量积
print(A.shape)
print(x.shape)
print(torch.mv(A, x))

# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
print(torch.abs(u).sum())
print(torch.norm(torch.ones((4, 9))))