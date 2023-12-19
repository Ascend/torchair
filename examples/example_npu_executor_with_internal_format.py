import os
import logging
import torch
import torch_npu

import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

logger.setLevel(logging.DEBUG)

print("current pid is ", os.getpid())
torch_npu.npu.set_device(0)

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 30)

    def forward(self, x, y):
        return self.fc(x), y + 1.0


in2 = torch.randn(2).to(torch.float16).npu()
ins = torch.randn(40, 10).to(torch.float16).npu()
model = Model().to(torch.float16).npu()

ins_nz = torch_npu.npu_format_cast(ins, 29)  # input to NZ
tng.experimental.inference.use_internal_format_weight(model)  # weight to NZ

eager_result = model(ins_nz, in2)
print("eager result: ", eager_result)

# check static graph
with torch.no_grad():
    static_graph_model = torch.compile(model, backend=npu_backend, dynamic=False)
    graph_result = static_graph_model(ins_nz, in2)
print("static graph result: ", graph_result)
assert (graph_result[0] - eager_result[0]).abs().max().item() <= 0.001

# check dynamic graph
with torch.no_grad():
    torch._dynamo.reset()
    dynamic_graph_model = torch.compile(model, backend=npu_backend, dynamic=True)
    torch._dynamo.mark_static(ins_nz)
    graph_result = dynamic_graph_model(ins_nz, in2)
print("dynamic graph result: ", graph_result)
assert (graph_result[0] - eager_result[0]).abs().max().item() <= 0.001
