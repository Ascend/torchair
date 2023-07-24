import os
import logging
from typing import Any, Dict, List, Tuple, Union
import torch.multiprocessing as mp
import torch
import torch_npu
import torch.distributed as dist
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig
import torchair as tng
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce

logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
config.debug.graph_dump.type = "pbtxt"

npu_backend = tng.get_npu_backend(compiler_config=config)


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        torch.distributed.all_reduce(x)
        return x


def example(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones([3], dtype=torch.int32).to("npu:" + str(rank))
    mod = MyModel()
    mod = mod.to("npu:" + str(rank))
    torch._dynamo.reset()
    opt_mod = torch.compile(
        mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(x)
    ori_result = mod(x)
    assert compile_result.equal(ori_result)


def main():
    world_size = 2
    mp.spawn(example,
             args=(world_size, ),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    main()
