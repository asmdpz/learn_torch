import torch
from d2l import torch as d2l

# 12.3 自动并行
# 12.3.1 基于GPU的并行计算
devices = d2l.try_all_gpus()