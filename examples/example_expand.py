from typing import Any, Dict, List, Tuple, Union
import functools
from torch._functorch.aot_autograd import aot_module_simplified
import torch

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c, d):
        a = a.expand([2, 3])
        a = a + b
        a = a.expand([1, 2, -1])
        a = torch.reshape(a, ([3, 2]))
        a = a - c
        a = a.expand([3, 3, 2])
        a = a + d
        a = torch.reshape(a, ([3, 6, 1]))
        a = a.expand([3, 6, 1])
        return a


model = Model().npu()
model = torch.compile(model, backend=npu_backend, dynamic=False)
in1 = torch.randn([3])
in2 = torch.randn([2, 3])
in3 = torch.randn([3, 2])
in4 = torch.randn([3, 3, 2])
out = model(in1, in2, in3, in4)
