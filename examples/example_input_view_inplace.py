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
config.experimental_config.keep_inference_input_mutations = True
npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dst_in, indices_in, src_in, dim_i):
        torch_npu.scatter_update_(dst_in, indices_in, src_in, dim_i)


dst = torch.ones(2, 1, 16, 128).float()
src = torch.randn(2, 1, 4, 64).float()
indices = torch.tensor([1, 1]).int().npu()
update_axis = -2

model = Model()
graph_model = torch.compile(model, backend=npu_backend, dynamic=False)


# eager result
dst_eager_all = dst.npu()
dst_eager = dst_eager_all[:, :, :, 1:65]
src_eager = src.npu()
model(dst_eager, indices, src_eager, update_axis)


# eager result
dst_graph_all = dst.npu()
dst_graph = dst_graph_all[:, :, :, 1:65]
src_graph = src.npu()
graph_result = graph_model(dst_graph, indices, src_graph, update_axis)

print("eager result: ", dst_eager_all)
print("graph result: ", dst_graph_all)

if not abs(dst_graph - dst_eager).sum().item() == 0: raise AssertionError
