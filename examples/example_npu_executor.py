
from typing import Any, Dict, List, Tuple, Union
from torch._functorch.aot_autograd import aot_module_simplified
import torch
import torch_npu
import functools

torch_npu.npu.set_device(1)

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
config.debug.graph_dump.type = "pbtxt"

npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.add(x, y*z)


model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=False)

in1 = torch.randn(4, 1).float().npu()
in2 = torch.randn(4, 4).float().npu()
in3 = torch.randn(4, 4).int().npu()

model(in1, in2, in3)

