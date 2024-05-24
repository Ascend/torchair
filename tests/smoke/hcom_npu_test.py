import os
import logging
import shutil
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed as dist
from torch.distributed import distributed_c10d
import torch.distributed.distributed_c10d as c10d
import torch_npu
import torchair
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig
import torchair.inference


class All2allsinge(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1):
        dist.all_to_all_single(output1, input1, output_split_sizes=[1, 1, 1, 1], input_split_sizes=[1, 1, 1, 1])
        return output1


def test_alltoall_single_dynamic(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    tensor_input = torch.arange(4) + rank * 4
    tensor_input = tensor_input.npu()
    tensor_output = torch.empty([4], dtype=torch.int64).npu()
    tensor_output_single = torch.empty([4], dtype=torch.int64).npu()

    model = All2allsinge().npu()
    config = CompilerConfig()
    
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    dist.all_to_all_single(tensor_output_single, tensor_input,
                           output_split_sizes=[1, 1, 1, 1], input_split_sizes=[1, 1, 1, 1])
    model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
    with torch.no_grad():
        tensor_output = model(tensor_input, tensor_output)
    assert tensor_output.equal(tensor_output_single)
    dist.destroy_process_group()


class AllToAllSingeSplitSize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1, input_split_sizes, output_split_sizes):
        dist.all_to_all_single(output1, input1, output_split_sizes=output_split_sizes,
                               input_split_sizes=input_split_sizes)
        return output1


def test_alltoall_single_dynamic_split_size(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    if rank == 0:
        input1 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).npu()
        output1 = torch.empty([9], dtype=torch.int64).npu()
        output1_single = torch.empty([9], dtype=torch.int64).npu()
        input_split_sizes = [2, 2, 1, 1]
        output_split_sizes = [2, 3, 2, 2]
    elif rank == 1:
        input1 = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int64).npu()
        output1 = torch.empty([7], dtype=torch.int64).npu()
        output1_single = torch.empty([7], dtype=torch.int64).npu()
        input_split_sizes = [3, 2, 2, 2]
        output_split_sizes = [2, 2, 1, 2]
    elif rank == 2:
        input1 = torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64).npu()
        output1 = torch.empty([6], dtype=torch.int64).npu()
        output1_single = torch.empty([6], dtype=torch.int64).npu()
        input_split_sizes = [2, 1, 1, 1]  
        output_split_sizes = [1, 2, 1, 2]
    elif rank == 3:
        input1 = torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.int64).npu()
        output1 = torch.empty([5], dtype=torch.int64).npu()
        output1_single = torch.empty([5], dtype=torch.int64).npu()
        input_split_sizes = [2, 2, 2, 1]
        output_split_sizes = [1, 2, 1, 1]

    model = AllToAllSingeSplitSize().npu()

    config = CompilerConfig()
    
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    output1_single = model(input1, output1_single, input_split_sizes, output_split_sizes)
    model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1, input_split_sizes, output_split_sizes)

    assert output1.equal(output1_single)
    dist.destroy_process_group()


def test_alltoall_single_static_split_size(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    if rank == 0:
        input1 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).npu()
        output1 = torch.empty([9], dtype=torch.int64).npu()
        output1_single = torch.empty([9], dtype=torch.int64).npu()
        input_split_sizes = [2, 2, 1, 1]
        output_split_sizes = [2, 3, 2, 2]
    elif rank == 1:
        input1 = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int64).npu()
        output1 = torch.empty([7], dtype=torch.int64).npu()
        output1_single = torch.empty([7], dtype=torch.int64).npu()
        input_split_sizes = [3, 2, 2, 2]
        output_split_sizes = [2, 2, 1, 2]
    elif rank == 2:
        input1 = torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64).npu()
        output1 = torch.empty([6], dtype=torch.int64).npu()
        output1_single = torch.empty([6], dtype=torch.int64).npu()
        input_split_sizes = [2, 1, 1, 1]  
        output_split_sizes = [1, 2, 1, 2]
    elif rank == 3:
        input1 = torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.int64).npu()
        output1 = torch.empty([5], dtype=torch.int64).npu()
        output1_single = torch.empty([5], dtype=torch.int64).npu()
        input_split_sizes = [2, 2, 2, 1]
        output_split_sizes = [1, 2, 1, 1]

    model = AllToAllSingeSplitSize().npu()

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    output1_single = model(input1, output1_single, input_split_sizes, output_split_sizes)
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1, input_split_sizes, output_split_sizes)
    assert output1.equal(output1_single)
    dist.destroy_process_group()


def test_alltoall_single_static(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    output1 = torch.empty([4], dtype=torch.int64).npu()
    output1_single = torch.empty([4], dtype=torch.int64).npu()

    model = All2allsinge().npu()

    config = CompilerConfig()
    output1_single = model(input1, output1_single)
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1)
    assert output1.equal(output1_single)
    dist.destroy_process_group()


class AllToAllASingeNoSplit(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1):
        dist.all_to_all_single(output1, input1)
        return output1


def test_alltoall_single_nosplit(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    output1 = torch.empty([4], dtype=torch.int64).npu()
    output1_single = torch.empty([4], dtype=torch.int64).npu()

    model = AllToAllASingeNoSplit().npu()
    config = CompilerConfig()
    output1_single = model(input1, output1_single)
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1)
    assert output1.equal(output1_single)
    dist.destroy_process_group()


def test_alltoall_single_nosplit_static(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    output1 = torch.empty([4], dtype=torch.int64).npu()
    output1_single = torch.empty([4], dtype=torch.int64).npu()

    model = AllToAllASingeNoSplit().npu()

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    output1_single = model(input1, output1_single)
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1)
    assert output1.equal(output1_single)
    dist.destroy_process_group()


class AllToAllSingeNoSplitInputOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1):
        input1 = input1 + 1
        dist.all_to_all_single(output1, input1)
        return output1 + 1


def test_alltoall_single_nosplit_static_inoutput(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    print("input: ", input1)
    output1 = torch.empty([4], dtype=torch.int64).npu()
    output1_single = torch.empty([4], dtype=torch.int64).npu()

    model = AllToAllSingeNoSplitInputOutput().npu()

    config = CompilerConfig()
    output1_single = model(input1, output1_single)
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1)
    assert output1.equal(output1_single)
    dist.destroy_process_group()


def test_alltoall_single_export(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    output1 = torch.empty([4], dtype=torch.int64).npu()

    model = AllToAllSingeNoSplitInputOutput().npu()
    torchair.dynamo_export(input1, output1, model=model, dynamic=True)
    dist.destroy_process_group()


class AllToAll(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1):
        dist.all_to_all(output1, input1)
        return output1


def test_alltoall(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    input1 = list(input1.chunk(4))
    output1 = torch.empty([4], dtype=torch.int64).npu()
    output1 = list(output1.chunk(4))
    output1_single = torch.empty([4], dtype=torch.int64).npu()
    output1_single = list(output1_single.chunk(4))
    model = AllToAll().npu()
    config = CompilerConfig()
    output1_single = model(input1, output1_single)
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1)
    for i, output_tensor in enumerate(output1):
        assert output_tensor.equal(output1_single[i])
    dist.destroy_process_group()


def test_alltoall2(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    input1 = list(input1.chunk(4))

    output1 = torch.empty([4], dtype=torch.int64).npu()
    output1 = list(output1.chunk(4))
    output1_single = torch.empty([4], dtype=torch.int64).npu()
    output1_single = list(output1_single.chunk(4))
    model = AllToAll().npu()
    config = CompilerConfig()
    
    output1_single = model(input1, output1_single)
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        output1 = model(input1, output1)
    for i, output_tensor in enumerate(output1):
        assert output_tensor.equal(output1_single[i])
    dist.destroy_process_group()


def test_alltoall3(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input1 = torch.arange(4) + rank * 4
    input1 = input1.npu()
    input1 = list(input1.chunk(4))

    output1 = torch.empty([4], dtype=torch.int64).npu()
    output1 = list(output1.chunk(4))

    model = AllToAll().npu()
    torchair.dynamo_export(input1, output1, model=model, dynamic=True)
    dist.destroy_process_group()


def test_alltoall4(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    input_list = [(torch.zeros(rank + 1, 1) + rank).float().npu() for i in range(world_size)]
    output_list = [torch.empty(i + 1, 1).float().npu() for i in range(world_size)]
    output_list_single = [torch.empty(i + 1, 1).float().npu() for i in range(world_size)]

    model = AllToAll().npu()
    config = CompilerConfig()
    output_list_single = model(input_list, output_list_single)
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        output_list = model(input_list, output_list)
    for i, output_tensor in enumerate(output_list):
        assert output_tensor.equal(output_list_single[i])
    dist.destroy_process_group()


def check_export_file_and_clean_env():
    assert os.path.exists("export_file")
    with open('export_file/rank_0/dynamo.pbtxt', 'r') as f:
        src = f.read()
    assert src.count("op: \"HcomAllToAllV\"") == 1
    shutil.rmtree("export_file")


class CacheHcomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_module = torchair.inference.cache_compile(self.prompt, dynamic=False)

    def inner_forward(self, x, y):
        ret = x + y
        torch.distributed.all_reduce(ret)
        return ret

    def forward(self, x, y):
        return self.cached_module(x, y)

    def prompt(self, x, y):
        return self.inner_forward(x, y)


def test_cache_allreduce(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    model = CacheHcomModel().npu()
    x = torch.ones(2, 2).npu()
    y = torch.ones(2, 2).npu()
    out = torch.ones(2, 2).npu() * 2 * world_size
    ret = model(x, y)
    assert out.equal(ret)
    dist.destroy_process_group()
    return


def check_cache_file_and_clean_env(path: str = ''):
    if not path:
        path = ".torchair_cache"
    assert os.path.exists(path)
    shutil.rmtree(path)


def mp():
    world_size = 4
    # =================  case 1 基本入图场景 动态图 + 单算子混跑 + split_sizes入参==================
    torch.multiprocessing.spawn(test_alltoall_single_dynamic, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 1 pass =============================", flush=True)
    # =================  case 2 基本入图场景 静态图 + 单算子混跑 + split_sizes入参==================
    torch.multiprocessing.spawn(test_alltoall_single_static, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 2 pass =============================", flush=True)
    # =================  case 3 基本入图场景 动态图 + 单算子混跑 + 无split_sizes入参==================
    torch.multiprocessing.spawn(test_alltoall_single_nosplit, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 3 pass =============================", flush=True)
    # =================  case 4 基本入图场景 静态图 + 单算子混跑 + 无split_sizes入参==================
    torch.multiprocessing.spawn(test_alltoall_single_nosplit_static, args=(world_size, ),
                                nprocs=world_size, join=True)
    print("==================case 4 pass =============================", flush=True)
    # =================  case 5 基本入图场景 静态图 + 单算子混跑 + 无split_sizes入参，不直连输入输入输出============
    torch.multiprocessing.spawn(test_alltoall_single_nosplit_static_inoutput,
                                args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 5 pass =============================", flush=True)
    # =================  case 6 基本入图场景 静态图 + 单算子混跑 + 无split_sizes入参，不直连输入输入输出 export======
    torch.multiprocessing.spawn(test_alltoall_single_export, args=(world_size, ), nprocs=world_size, join=True)
    check_export_file_and_clean_env()
    print("==================case 6 pass =============================", flush=True)
    # =================  case 7 动态图 + split_sizes入参不等分==================
    torch.multiprocessing.spawn(test_alltoall_single_dynamic_split_size, args=(world_size, ),
                                nprocs=world_size, join=True)
    print("==================case 7 pass =============================", flush=True)
    # =================  case 8 静态图 + split_sizes入参不等分==================
    torch.multiprocessing.spawn(test_alltoall_single_static_split_size, args=(world_size, ),
                                nprocs=world_size, join=True)
    print("==================case 8 pass =============================", flush=True)
    # =================  case 9 动态图 + all2all基本用例==================
    torch.multiprocessing.spawn(test_alltoall, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 9 pass =============================", flush=True)
    # =================  case 10 动态图 + all2all 单算子图混跑==================
    torch.multiprocessing.spawn(test_alltoall2, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 10 pass =============================", flush=True)

    # =================  case 11 动态图 + all2all tensor不等分==================
    torch.multiprocessing.spawn(test_alltoall4, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 11 pass =============================", flush=True)

    # =================  case 12 动态图 + all2all export==================
    torch.multiprocessing.spawn(test_alltoall3, args=(world_size, ), nprocs=world_size, join=True)
    check_export_file_and_clean_env()
    print("==================case 12 pass =============================", flush=True)
    print("==================all case all_to_all pass =================", flush=True)

    # =================  case 13 creat cache + hcom ==================
    torch.multiprocessing.spawn(test_cache_allreduce, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 13 pass =============================", flush=True)
    # =================  case 14 use cache + hcom ==================
    torch.multiprocessing.spawn(test_cache_allreduce, args=(world_size, ), nprocs=world_size, join=True)
    print("==================case 14 pass =============================", flush=True)
    check_cache_file_and_clean_env()

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"
    mp()
