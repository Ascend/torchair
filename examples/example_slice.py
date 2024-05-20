
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

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, 1:2]


model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=True)
model(torch.randn(3, 2))
