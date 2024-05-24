
from typing import Any, Dict, List, Tuple, Union
from torch._functorch.aot_autograd import aot_module_simplified
import torch
import functools

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"

npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)


model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=True)
x = torch.randn(512, 1024, 1024)
y = torch.randn(512, 1024, 1024)
for i in range(1000):
    model(x, y)
