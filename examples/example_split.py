
from typing import Any, Dict, List, Tuple, Union
from torch._functorch.aot_autograd import aot_module_simplified
import torch
import functools

import torchair as tng
from torchair._ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"

npu_backend = tng.get_npu_backend(compiler_config=config)


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        x = torch.cat([torch.ones(x.size()), torch.ones(y.size())])
        x = torch.ones(x.size())
        x = torch.split(x, z, dim=0)
        return x[-1], x[0]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


model = Model2()
model = torch.compile(model, backend=npu_backend, dynamic=True)
model(torch.randn(2, 2), torch.randn(2, 2), [2, 2])
model(torch.randn(3, 3), torch.randn(3, 3), [3, 3])
model(torch.randn(4, 4), torch.randn(4, 4), [4, 4])
