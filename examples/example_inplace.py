import torch
import torchair as tng
import torch_npu
from torchair.configs.compiler_config import CompilerConfig

import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
config.experimental_config.keep_inference_input_mutations = True

npu_backend = tng.get_npu_backend(compiler_config=config)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x[1].add_(y)

model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=False)

x = torch.ones(2, 2).npu()
y = torch.ones(2).npu()
model(x, y)
print(x)
