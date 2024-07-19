import os
import torch

import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce


class AllReduceSingeGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x + y
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        return x


def example(rank, world_size):
    torch.npu.set_device(rank)
    torch.distributed.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones([2, 2], dtype=torch.int32).to("npu:"+str(rank))
    y = torch.ones([2, 2], dtype=torch.int32).to("npu:"+str(rank))
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(AllReduceSingeGroup().to("npu:"+str(rank)), backend=npu_backend, dynamic=False)
    out = torch.ones([2, 2], dtype=torch.int32).npu() * 2 * world_size
    ret = model(x, y)
    assert out.equal(ret)
    torch.distributed.destroy_process_group()


def mp():
    world_size = 2
    torch.multiprocessing.spawn(example, args=(world_size, ), nprocs=world_size, join=True)

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"    
    mp()
