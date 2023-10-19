import torch
import torch_npu
import os

print("current pid is ", os.getpid())
torch_npu.npu.set_device(1)

import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import logging
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.add(x, y*z)


model = Model()

in1 = torch.randn(4, 1).float().npu()
in2 = torch.randn(4, 4).float().npu()
in3 = torch.randn(4, 4).int().npu()

eager_result = model(in1, in2, in3)

model = torch.compile(model, backend=npu_backend, dynamic=True)
graph_result = model(in1, in2, in3)

print("eager result: ", eager_result)
print("graph result: ", graph_result)
