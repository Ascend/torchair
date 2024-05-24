import os
import shutil
import logging
import torch.multiprocessing as mp
import torch
import torch.distributed._functional_collectives as funcol
import torch_npu
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce import functional_collectives_context

os.environ['TNG_LOG_LEVEL'] = '0'
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)


class AllReduceSingeGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        torch.distributed.all_reduce(x)
        x = x + 1
        return x


class AllReduceMultiGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test1')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test2')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test3')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test1') # 重复的group case
        x = x + 1
        return x


class DistReduceScatterTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output):
        from torch.distributed.distributed_c10d import _world
        # 必须要带group参数
        torch.distributed.reduce_scatter_tensor(output, x, group=_world.default_pg)
        return x


class FuncolReduceScatterTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        from torch.distributed.distributed_c10d import _world
        out = funcol.reduce_scatter_tensor(x, 'sum', scatter_dim=-1, group=_world.default_pg)
        return out


def example(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    torch.distributed.init_process_group("hccl", rank=rank, world_size=world_size)

    print("=======================test case 1===================")
    x = torch.ones([3], dtype=torch.int32).to("npu:" + str(rank))
    y = torch.ones([3], dtype=torch.int32).to("npu:" + str(rank))
    z = torch.ones([3], dtype=torch.int32).to("npu:" + str(rank))
    mod1 = AllReduceSingeGroup()
    mod1 = mod1.to("npu:" + str(rank))
    torch._dynamo.reset()
    with functional_collectives_context():
        opt_mod1 = torch.compile(mod1, dynamic=False, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod1(x)
        print("AllReduceSingeGroup dynamic=False compile_result: ", compile_result)
        opt_mod1_true = torch.compile(mod1, dynamic=True, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod1_true(z)
        print("AllReduceSingeGroup dynamic=True compile_result: ", compile_result)
    ori_result = mod1(y)
    print("AllReduceSingeGroup ori_result: ", ori_result)

    print("=======================test case 2===================")
    x = torch.ones([3], dtype=torch.int32).to("npu:" + str(rank))
    y = torch.ones([3], dtype=torch.int32).to("npu:" + str(rank))
    z = torch.ones([3], dtype=torch.int32).to("npu:" + str(rank))
    mod2 = AllReduceMultiGroup()
    mod2 = mod2.to("npu:" + str(rank))
    torch._dynamo.reset()
    with functional_collectives_context():
        opt_mod2 = torch.compile(mod2, dynamic=False, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod2(x)
        print("AllReduceMultiGroup dynamic=False compile_result: ", compile_result)
        opt_mod2_ture = torch.compile(mod2, dynamic=True, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod2_ture(z)
        print("AllReduceMultiGroup dynamic=True compile_result: ", compile_result)
    ori_result = mod2(y)
    print("AllReduceMultiGroup ori_result: ", ori_result)

    print("=======================test case 3===================")
    mod3 = DistReduceScatterTensor()
    mod3 = mod3.to("npu:" + str(rank))
    xx = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("npu:" + str(rank))
    yy = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("npu:" + str(rank))
    zz = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("npu:" + str(rank))
    output1 = torch.empty([2], dtype=torch.int32).to("npu:" + str(rank))
    output2 = torch.empty([2], dtype=torch.int32).to("npu:" + str(rank))
    output3 = torch.empty([2], dtype=torch.int32).to("npu:" + str(rank))
    torch._dynamo.reset()

    opt_mod3 = torch.compile(mod3, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod3(xx, output1)
    print("DistReduceScatterTensor dynamic=False compile_result: ", compile_result)
    opt_mod3_true = torch.compile(mod3, dynamic=True, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod3_true(yy, output2)
    print("DistReduceScatterTensor dynamic=True compile_result: ", compile_result)
    ori_result = mod3(zz, output3)
    print("DistReduceScatterTensor ori_result: ", ori_result)

    print("=======================test case 4===================")
    mod4 = FuncolReduceScatterTensor()
    mod4 = mod4.to("npu:" + str(rank))
    torch._dynamo.reset()
    xx = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("npu:" + str(rank))
    yy = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("npu:" + str(rank))
    zz = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("npu:" + str(rank))
    opt_mod4 = torch.compile(mod4, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod4(xx)
    print("FuncolReduceScatterTensor dynamic=False compile_result: ", compile_result)
    opt_mod4_true = torch.compile(mod4, dynamic=True, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod4_true(yy)
    print("FuncolReduceScatterTensor dynamic=True compile_result: ", compile_result)
    ori_result = mod4(zz)
    print("FuncolReduceScatterTensor ori_result: ", ori_result)


def main():
    world_size = 2
    mp.spawn(example,
             args=(world_size, ),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29516"
    main()