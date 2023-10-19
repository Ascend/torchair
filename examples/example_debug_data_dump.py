
from typing import Any, Dict, List, Tuple, Union
from torch._functorch.aot_autograd import aot_module_simplified
import torch
import functools
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.debug.data_dump.type = "npy"

npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.add(x, x)


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.split(x, 2)


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.concat([x, x])


class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.add(x, 1)
        return torch.sub(x, 2)


# 常规dump
model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=True)
x = torch.randn(2, 2)
model(x)
model(x)

# Dump list输出
model = Model1()
model = torch.compile(model, backend=npu_backend, dynamic=True)
x = torch.randn(4, 2)
model(x)
model(x)

# Dump list输入
model = Model2()
model = torch.compile(model, backend=npu_backend, dynamic=True)
x = torch.randn(2, 2)
model(x)
model(x)

# 只Dump指定node
config = CompilerConfig()
config.debug.data_dump.type = "npy"

# 自定义node的过滤规则，node.target为节点类型，node.name为节点名称，node.stack_trace为节点调用栈
def filter(node):
    if 'add' not in str(node.target):
        return None
    return node


config.debug.data_dump.filter = filter

npu_backend = tng.get_npu_backend(compiler_config=config)
model = Model3()
model = torch.compile(model, backend=npu_backend, dynamic=True)
x = torch.randn(2, 2)
model(x)
model(x)
