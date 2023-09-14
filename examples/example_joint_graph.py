import os
import logging

import torch
import torch_npu

import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.configs.aot_config import AotConfig
from torchair.core.utils import logger

logger.setLevel(logging.DEBUG)
print("current pid is ", os.getpid())
torch_npu.npu.set_device(0)

config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
config.debug.graph_dump.type = "pbtxt"

aot_config = AotConfig()
aot_config.enable_joint_graph = True
aot_config.output_loss_index = 0

npu_backend = tng.get_npu_backend(compiler_config=config, aot_config=aot_config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 256, 3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        output = self.relu(x)
        loss = output.sum([0, 1, 2, 3])
        return loss


model = Model().npu()

input_tensor = torch.randn(4, 64, 32, 32).float().npu()

model = torch.compile(model, backend=npu_backend, dynamic=False)

with torch.npu.amp.autocast(True):
    graph_result = model(input_tensor)

print("conv.weight.grad", model.conv.weight.grad)