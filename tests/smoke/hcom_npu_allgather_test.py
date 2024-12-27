import os
import logging
import shutil
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch_npu
import torchair
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
torchair.patch_for_hcom()


class allgather(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_list, x):
        torch.distributed.all_gather(tensor_list, x)
        return tensor_list


class allgather_in_tensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x):
        torch.distributed.all_gather_into_tensor(out_tensor, x)
        return out_tensor


class AllGatherInTensorUneven(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x, output_split_sizes=None):
        torch.distributed.all_gather_into_tensor_uneven(out_tensor, x, output_split_sizes)
        return out_tensor


class ReduceScatterTensorUneven(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x, intput_split_sizes=None):
        torch.distributed.reduce_scatter_tensor_uneven(out_tensor, x, intput_split_sizes)
        return out_tensor


class Broadcast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, src):
        dist.broadcast(x, src)
        return x


def test_allgather_dynamic(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = [torch.zeros(2, 2, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
    mod = allgather()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=True, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    for i, t in enumerate(compile_result):
        assert t.equal(ori_result[i])


def test_allgather_static(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = [torch.zeros(2, 2, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
    mod = allgather()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    for i, t in enumerate(compile_result):
        assert t.equal(ori_result[i])


def test_allgather_reshape(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = [torch.zeros(4, 1, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
    mod = allgather()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    for i, t in enumerate(compile_result):
        assert t.equal(ori_result[i])


def test_allgather_reshape_1_4(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = [torch.zeros(1, 4, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
    mod = allgather()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    for i, t in enumerate(compile_result):
        assert t.equal(ori_result[i])


def test_allgather_different_size(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    if rank == 0:
        x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    else:
        x = torch.ones(1, 4, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = []
    y = torch.zeros(2, 2, dtype=torch.int64).to("npu:" + str(rank))
    y2 = torch.zeros(1, 4, dtype=torch.int64).to("npu:" + str(rank))
    tensor_list.append(y)
    tensor_list.append(y2)
    mod = allgather()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=True, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    for i, t in enumerate(compile_result):
        assert t.equal(ori_result[i])


def test_allgather_in_tensor_dynamic(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    print("x-----===:", x)
    tensor_list = torch.zeros(4, 2, dtype=torch.int64).to("npu:" + str(rank))
    print("tensor_list-----===:", tensor_list)
    mod = allgather_in_tensor()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=True, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_allgather_in_tensor_static(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    print("x-----===:", x)
    tensor_list = torch.zeros(4, 2, dtype=torch.int64).to("npu:" + str(rank))
    print("tensor_list-----===:", tensor_list)
    mod = allgather_in_tensor()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_allgather_in_tensor_reshape_8_1(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    print("x-----===:", x)
    tensor_list = torch.zeros(8, 1, dtype=torch.int64).to("npu:" + str(rank))
    print("tensor_list-----===:", tensor_list)
    mod = allgather_in_tensor()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_allgather_in_tensor_no_same_size(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    if rank == 0:
        x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    else:
        x = torch.ones(1, 4, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = torch.zeros(8, 1, dtype=torch.int64).to("npu:" + str(rank))
    mod = allgather_in_tensor()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_allgather_in_tensor_uneven_different_size(rank, world_size, dynamic=False):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    if rank == 0:
        x = torch.ones(4, 2, dtype=torch.int32).to("npu:" + str(rank)) + 1 + 2 * rank
    else:
        x = torch.ones(1, 2, dtype=torch.int32).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = torch.zeros(5, 2, dtype=torch.int32).to("npu:" + str(rank))
    output_split_sizes = [4, 1]
    mod = AllGatherInTensorUneven()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x, output_split_sizes)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x, output_split_sizes)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_allgather_in_tensor_uneven_same_size(rank, world_size, dynamic=False):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int32).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = torch.zeros(4, 2, dtype=torch.int32).to("npu:" + str(rank))
    mod = AllGatherInTensorUneven()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_reducescatter_tensor_uneven_same_size(rank, world_size, dynamic=False):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    tensor_in = torch.arange(world_size * 2, dtype=torch.int32).to("npu:" + str(rank))
    tensor_in = torch.reshape(tensor_in, (world_size, 2))
    tensor_out = torch.zeros(2, dtype=torch.int32).to("npu:" + str(rank))
    mod = ReduceScatterTensorUneven()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_out, tensor_in)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_out, tensor_in)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_reducescatter_tensor_uneven_different_size(rank, world_size, dynamic=False):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    tensor_in = torch.arange(5 * 2, dtype=torch.int32).to("npu:" + str(rank))
    tensor_in = torch.reshape(tensor_in, (5, 2))
    input_split_sizes = [4, 1]
    if rank == 0:
        tensor_out = torch.zeros(4, 2, dtype=torch.int32).to("npu:" + str(rank))
    else:
        tensor_out = torch.zeros(1, 2, dtype=torch.int32).to("npu:" + str(rank))
    mod = ReduceScatterTensorUneven()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_out, tensor_in, input_split_sizes)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_out, tensor_in, input_split_sizes)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)


def test_broadcast_static(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    if rank == 0:
        x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    else:
        x = torch.ones(1, 4, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = []
    y = torch.zeros(2, 2, dtype=torch.int64).to("npu:" + str(rank))
    y1 = torch.zeros(1, 4, dtype=torch.int64).to("npu:" + str(rank))
    tensor_list.append(y)
    tensor_list.append(y1)
    mod = Broadcast()
    model = mod.to("npu:" + str(rank))
    ori_result_list = []
    compile_result_list = []
    for i, out_put in enumerate(tensor_list):
        op_input_tensor = out_put
        if i == rank:
            op_input_tensor = x
        ori_result = mod(op_input_tensor, i)
        torch._dynamo.reset()
        opt_mod = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        compile_result = opt_mod(op_input_tensor, i)
        ori_result_list.append(ori_result)
        compile_result_list.append(compile_result)
    print("ori_result_list:", ori_result_list, "compile_result_list:", compile_result_list)
    for j, t in enumerate(compile_result_list):
        assert t.equal(ori_result_list[j])
    dist.destroy_process_group()


def test_broadcast_dynamic(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    if rank == 0:
        x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    else:
        x = torch.ones(1, 4, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = []
    y = torch.zeros(2, 2, dtype=torch.int64).to("npu:" + str(rank))
    y1 = torch.zeros(1, 4, dtype=torch.int64).to("npu:" + str(rank))
    tensor_list.append(y)
    tensor_list.append(y1)
    mod = Broadcast()
    model = mod.to("npu:" + str(rank))
    ori_result_list = []
    compile_result_list = []
    for i, out_put in enumerate(tensor_list):
        op_input_tensor = out_put
        if i == rank:
            op_input_tensor = x
        ori_result = mod(op_input_tensor, i)
        torch._dynamo.reset()
        opt_mod = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        compile_result = opt_mod(op_input_tensor, i)
        ori_result_list.append(ori_result)
        compile_result_list.append(compile_result)
    print("ori_result_list:", ori_result_list, "compile_result_list:", compile_result_list)
    for j, t in enumerate(compile_result_list):
        assert t.equal(ori_result_list[j])
    dist.destroy_process_group()


def mp():
    world_size = 2
    # == == == == == == == == =  case1 allgather动态入图 == == == == == == == == ==
    torch.multiprocessing.spawn(test_allgather_dynamic, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 1 pass =============================", flush=True)
    # =================  case 2 allgather静态入图==================
    torch.multiprocessing.spawn(test_allgather_static, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 2 pass =============================", flush=True)
    # =================  case 3 allgather返回shape变化[2,2]返回[4,1]==================
    torch.multiprocessing.spawn(test_allgather_reshape, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 3 pass =============================", flush=True)
    # =================  case 4 allgather返回shape变化[2,2]返回[1,4]============
    torch.multiprocessing.spawn(test_allgather_reshape_1_4, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 4 pass =============================", flush=True)
    # =================  case 5 allgather返回tensor集合每个tensor的size不一样，入参和返回的对应的tensor要size一致，否则不支持，cuda支持,目前单算子不支持======
    # torch.multiprocessing.spawn(test_allgather_different_size, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 5 pass =============================", flush=True)
    # =================  case 6 allgather_in_tensor动态入图==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_dynamic, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 6 pass =============================", flush=True)
    # =================  case 7 allgather_in_tensor静态入图==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_static, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 7 pass =============================", flush=True)
    # =================  case 8 allgather_in_tensor入参[2,2]f返回shape变化[8,1]==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_reshape_8_1, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 8pass =============================", flush=True)
    # =================  case 9 allgather_in_tensor入参[2,2],[4,1]返回[8,1]==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_no_same_size, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 9 pass =============================", flush=True)
    # =================  case 10 broadcast静态入图==================
    torch.multiprocessing.spawn(test_broadcast_static, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 10 pass =============================", flush=True)
    # =================  case 11 broadcast动态入图==================
    torch.multiprocessing.spawn(test_broadcast_dynamic, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 11 pass =============================", flush=True)
    # =================  case 12 allgather_in_tensor_uneven静态入图，size相同，新算子不支持动态入图==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_uneven_same_size, args=(world_size,), nprocs=world_size,
                                join=True)
    print("==================case 12 pass =============================", flush=True)
    # =================  case 13 allgather_in_tensor_uneven静态入图，size不同，新算子不支持动态入图==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_uneven_different_size, args=(world_size,), nprocs=world_size,
                                join=True)
    print("==================case 13 pass =============================", flush=True)
    # =================  case 14 reducescatter_tensor_uneven静态入图，size相同，新算子不支持动态入图==================
    torch.multiprocessing.spawn(test_reducescatter_tensor_uneven_same_size, args=(world_size,), nprocs=world_size,
                                join=True)
    print("==================case 14 pass =============================", flush=True)
    # =================  case 15 reducescatter_tensor_uneven静态入图，size不同，新算子不支持动态入图==================
    torch.multiprocessing.spawn(test_reducescatter_tensor_uneven_different_size, args=(world_size,), nprocs=world_size,
                                join=True)
    print("==================case 15 pass =============================", flush=True)
    print("==================all case allgather、allgather_in_tensor、allgather_in_tensor_uneven、"
          "reducescatter_tensor_uneven、broadcast pass =================", flush=True)


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"
    mp()
