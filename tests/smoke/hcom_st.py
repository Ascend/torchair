import os
import unittest
from unittest.mock import Mock, patch
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig


config = CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)


class AllToAllSingeSplitSize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1, input_split_sizes, output_split_sizes):
        dist.all_to_all_single(output1, input1, output_split_sizes=output_split_sizes,
                               input_split_sizes=input_split_sizes)
        return output1


class All2allsinge(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1):
        dist.all_to_all_single(output1, input1)
        return output1


class AllToAllSingeNoSplitInputOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1):
        input1 = input1 + 1
        dist.all_to_all_single(output1, input1)
        return output1 + 1


class AllReduceSingeGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        handle = torch.distributed.all_reduce(
            x, op=torch.distributed.ReduceOp.SUM, async_op=True)
        handle.wait()
        x = x + y
        return x


class AllToAll(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, output1):
        dist.all_to_all(output1, input1)
        return output1


class AllGather(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_list, x):
        torch.distributed.all_gather(tensor_list, x)
        return tensor_list


class AllGatherInTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x):
        torch.distributed.all_gather_into_tensor(
            out_tensor, x, dist.group.WORLD)
        torch.distributed.all_gather_into_tensor(out_tensor, x)
        return out_tensor


class AllReduce(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x + y
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        return x


class ReduceScatterTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, input_tensor):
        if torch.__version__ >= '2.3.1':
            torch.distributed.reduce_scatter_tensor(out_tensor, input_tensor)
        else:
            # 2.1版本原生有bug, 且plugin修复成本过高，因此放弃修复，选择资料提示用户
            torch.distributed.reduce_scatter_tensor(
                out_tensor, input_tensor, group=dist.group.WORLD)
        return out_tensor


class AllGatherInTensorUneven(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x, output_split_sizes=None):
        torch_npu.distributed.all_gather_into_tensor_uneven(
            out_tensor, x, output_split_sizes)
        return out_tensor


class ReduceScatterTensorUneven(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x, intput_split_sizes=None):
        torch_npu.distributed.reduce_scatter_tensor_uneven(
            out_tensor, x, intput_split_sizes)
        return out_tensor


class Broadcast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, src):
        dist.broadcast(x, src)
        return x


class HcomTest(unittest.TestCase):
    @classmethod
    def _init_dist_hccl_without_patch(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29510'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(
            backend='hccl', world_size=world_size, rank=rank)

    @classmethod
    def _init_dist_hccl_with_patch(cls, rank, world_size):
        torchair.patch_for_hcom()
        cls._init_dist_hccl_without_patch(rank, world_size)
        return dist

    @classmethod
    def _test_all_to_all_single_compile(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        tensor_input = torch.arange(4) + rank * 4
        tensor_input = tensor_input.npu()
        tensor_output = torch.empty([4], dtype=torch.int64).npu()
        tensor_output_single = torch.empty([4], dtype=torch.int64).npu()
        model = All2allsinge().npu()
        dist.all_to_all_single(tensor_output_single, tensor_input)
        model = torch.compile(model, backend=npu_backend,
                              dynamic=dynamic, fullgraph=True)
        with torch.no_grad():
            tensor_output = model(tensor_input, tensor_output)
        results.append(tensor_output.equal(tensor_output_single))
        dist.destroy_process_group()

    @classmethod
    def _test_all_to_all_single_split_size_compile(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        if rank == 0:
            input1 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).npu()
            output1 = torch.empty([9], dtype=torch.int64).npu()
            output1_single = torch.empty([9], dtype=torch.int64).npu()
            input_split_sizes = [2, 2, 1, 1]
            output_split_sizes = [2, 3, 2, 2]
        elif rank == 1:
            input1 = torch.tensor(
                [10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int64).npu()
            output1 = torch.empty([7], dtype=torch.int64).npu()
            output1_single = torch.empty([7], dtype=torch.int64).npu()
            input_split_sizes = [3, 2, 2, 2]
            output_split_sizes = [2, 2, 1, 2]
        elif rank == 2:
            input1 = torch.tensor([20, 21, 22, 23, 24],
                                  dtype=torch.int64).npu()
            output1 = torch.empty([6], dtype=torch.int64).npu()
            output1_single = torch.empty([6], dtype=torch.int64).npu()
            input_split_sizes = [2, 1, 1, 1]
            output_split_sizes = [1, 2, 1, 2]
        elif rank == 3:
            input1 = torch.tensor(
                [30, 31, 32, 33, 34, 35, 36], dtype=torch.int64).npu()
            output1 = torch.empty([5], dtype=torch.int64).npu()
            output1_single = torch.empty([5], dtype=torch.int64).npu()
            input_split_sizes = [2, 2, 2, 1]
            output_split_sizes = [1, 2, 1, 1]

        model = AllToAllSingeSplitSize().npu()
        output1_single = model(input1, output1_single,
                               input_split_sizes, output_split_sizes)
        model = torch.compile(model, backend=npu_backend,
                              dynamic=dynamic, fullgraph=True)
        with torch.no_grad():
            output1 = model(input1, output1, input_split_sizes,
                            output_split_sizes)
        results.append(output1.equal(output1_single))
        dist.destroy_process_group()

    @classmethod
    def _test_all_to_all_single_dims_compile(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        input1 = torch.randn(768, 2048).npu()
        input1 = input1.npu()
        output1 = torch.randn(768, 2048).npu()
        output1_single = torch.randn(768, 2048).npu()
        model = All2allsinge().npu()
        output1_single = model(input1, output1_single)
        model = torch.compile(model, backend=npu_backend,
                              dynamic=dynamic, fullgraph=True)
        with torch.no_grad():
            output1 = model(input1, output1)
        results.append(output1.equal(output1_single))
        dist.destroy_process_group()

    @classmethod
    def _test_all_to_all_single_fuse_add_compile(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        input1 = torch.arange(4) + rank * 4
        input1 = input1.npu()
        output1 = torch.empty([4], dtype=torch.int64).npu()
        output1_single = torch.empty([4], dtype=torch.int64).npu()
        model = AllToAllSingeNoSplitInputOutput().npu()
        output1_single = model(input1, output1_single)
        model = torch.compile(model, backend=npu_backend,
                              dynamic=dynamic, fullgraph=True)
        with torch.no_grad():
            output1 = model(input1, output1)
        results.append(output1.equal(output1_single))
        dist.destroy_process_group()

    @classmethod
    def _test_all_to_all_single_export(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        input1 = torch.arange(4) + rank * 4
        input1 = input1.npu()
        output1 = torch.empty([4], dtype=torch.int64).npu()
        model = AllToAllSingeNoSplitInputOutput().npu()
        torchair.dynamo_export(input1, output1, model=model, dynamic=dynamic)

        def check_export_file_and_clean_env():
            results.append(os.path.exists("export_file"))
            file_path = 'export_file/rank_' + str(rank) + '/dynamo.pbtxt'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist")
            with open(file_path, 'r') as f:
                src = f.read()
            results.append(src.count("op: \"HcomAllToAllV\"") == 1)

        check_export_file_and_clean_env()
        dist.destroy_process_group()

    @classmethod
    def _test_eager_allreduce_async(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones([2, 2], dtype=torch.int32).to("npu:" + str(rank))
        y = torch.ones([2, 2], dtype=torch.int32).to("npu:" + str(rank))
        model = AllReduceSingeGroup().to("npu:" + str(rank))
        out = torch.ones([2, 2], dtype=torch.int32).to(
            "npu:" + str(rank)) * world_size + y
        ret = model(x, y)
        results.append(out.equal(ret))
        dist.destroy_process_group()

    @classmethod
    def _test_all_to_all(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        input1 = torch.arange(4) + rank * 4
        input1 = input1.npu()
        input1 = list(input1.chunk(4))
        output1 = torch.empty([4], dtype=torch.int64).npu()
        output1 = list(output1.chunk(4))
        output1_single = torch.empty([4], dtype=torch.int64).npu()
        output1_single = list(output1_single.chunk(4))
        model = AllToAll().npu()
        output1_single = model(input1, output1_single)
        model = torch.compile(model, backend=npu_backend,
                              dynamic=dynamic, fullgraph=True)
        with torch.no_grad():
            output1 = model(input1, output1)
        for i, output_tensor in enumerate(output1):
            results.append(output_tensor.equal(output1_single[i]))
        dist.destroy_process_group()

    @classmethod
    def _test_all_to_all(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        input1 = torch.arange(4) + rank * 4
        input1 = input1.npu()
        input1 = list(input1.chunk(4))
        output1 = torch.empty([4], dtype=torch.int64).npu()
        output1 = list(output1.chunk(4))
        output1_single = torch.empty([4], dtype=torch.int64).npu()
        output1_single = list(output1_single.chunk(4))
        model = AllToAll().npu()
        output1_single = model(input1, output1_single)
        model = torch.compile(model, backend=npu_backend,
                              dynamic=dynamic, fullgraph=True)
        with torch.no_grad():
            output1 = model(input1, output1)
        for i, output_tensor in enumerate(output1):
            results.append(output_tensor.equal(output1_single[i]))
        dist.destroy_process_group()

    @classmethod
    def _test_all_to_all_different_size(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        input_list = [(torch.zeros(rank + 1, 1) + rank).float().npu()
                      for i in range(world_size)]
        output_list = [torch.empty(i + 1, 1).float().npu()
                       for i in range(world_size)]
        output_list_single = [torch.empty(
            i + 1, 1).float().npu() for i in range(world_size)]
        model = AllToAll().npu()
        output_list_single = model(input_list, output_list_single)
        model = torch.compile(model, backend=npu_backend,
                              dynamic=False, fullgraph=True)
        with torch.no_grad():
            output_list = model(input_list, output_list)
        for i, output_tensor in enumerate(output_list):
            results.append(output_tensor.equal(output_list_single[i]))
        dist.destroy_process_group()

    @classmethod
    def _test_all_gather(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones(2, 2, dtype=torch.int64).to(
            "npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = [torch.zeros(2, 2, dtype=torch.int64).to(
            "npu:" + str(rank)) for _ in range(2)]
        mod = AllGather()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x)
        opt_mod = torch.compile(mod, dynamic=dynamic,
                                fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x)
        for i, t in enumerate(compile_result):
            results.append(t.equal(ori_result[i]))
        dist.destroy_process_group()

    @classmethod
    def _test_allreduce(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones([2, 2], dtype=torch.int32).to("npu:" + str(rank))
        y = torch.ones([2, 2], dtype=torch.int32).to("npu:" + str(rank))
        model = torch.compile(AllReduce().to("npu:" + str(rank)),
                              backend=npu_backend, dynamic=dynamic, fullgraph=True)
        out = torch.ones([2, 2], dtype=torch.int32).npu() * 2 * world_size
        ret = model(x, y)
        results.append(out.equal(ret))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_into_tensor(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones(2, 2, dtype=torch.int64).to(
            "npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = torch.zeros(
            4, 2, dtype=torch.int64).to("npu:" + str(rank))
        mod = AllGatherInTensor()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x)
        opt_mod = torch.compile(mod, dynamic=dynamic,
                                fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_reduce_scatter_tensor(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        input_tensor = torch.arange(
            world_size * 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
        output_tensor = torch.zeros(
            2, dtype=torch.int64).to("npu:" + str(rank))
        mod = ReduceScatterTensor()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(output_tensor, input_tensor)
        opt_mod = torch.compile(mod, dynamic=dynamic,
                                fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(output_tensor, input_tensor)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_broadcast(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
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
            opt_mod = torch.compile(model, backend=npu_backend, dynamic=dynamic, fullgraph=True)
            compile_result = opt_mod(op_input_tensor, i)
            ori_result_list.append(ori_result)
            compile_result_list.append(compile_result)
        for j, t in enumerate(compile_result_list):
            results.append(t.equal(ori_result_list[j]))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_in_tensor_uneven_same_size(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones(2, 2, dtype=torch.int32).to("npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = torch.zeros(4, 2, dtype=torch.int32).to("npu:" + str(rank))
        mod = AllGatherInTensorUneven()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_in_tensor_uneven_different_size(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        if rank == 0:
            x = torch.ones(4, 2, dtype=torch.int32).to("npu:" + str(rank)) + 1 + 2 * rank
        else:
            x = torch.ones(1, 2, dtype=torch.int32).to("npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = torch.zeros(5, 2, dtype=torch.int32).to("npu:" + str(rank))
        output_split_sizes = [4, 1]
        mod = AllGatherInTensorUneven()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x, output_split_sizes)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x, output_split_sizes)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_reducescatter_tensor_uneven_same_size(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        tensor_in = torch.arange(world_size * 2, dtype=torch.int32).to("npu:" + str(rank))
        tensor_in = torch.reshape(tensor_in, (world_size, 2))
        tensor_out = torch.zeros(1, 2, dtype=torch.int32).to("npu:" + str(rank))
        mod = ReduceScatterTensorUneven()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_out, tensor_in)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_out, tensor_in)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_reducescatter_tensor_uneven_different_size(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
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
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_out, tensor_in, input_split_sizes)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_reshape(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = [torch.zeros(4, 1, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
        mod = AllGather()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x)
        for i, t in enumerate(compile_result):
            results.append(t.equal(ori_result[i]))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_reshape_1_4(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = [torch.zeros(1, 4, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
        mod = AllGather()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x)
        for i, t in enumerate(compile_result):
            results.append(t.equal(ori_result[i]))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_in_tensor_reshape_8_1(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = torch.zeros(8, 1, dtype=torch.int64).to("npu:" + str(rank))
        mod = AllGatherInTensor()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_in_tensor_no_same_size(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        if rank == 0:
            x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
        else:
            x = torch.ones(1, 4, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
        tensor_list = torch.zeros(8, 1, dtype=torch.int64).to("npu:" + str(rank))
        mod = AllGatherInTensor()
        mod = mod.to("npu:" + str(rank))
        ori_result = mod(tensor_list, x)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_list, x)
        results.append(ori_result.equal(compile_result))
        dist.destroy_process_group()

    @classmethod
    def _test_allgather_in_tensor_check_shape(cls, rank, world_size, init_pg, dynamic, results):
        torch.npu.set_device(rank)
        init_pg(rank, world_size) 
        device = "npu:" + str(rank)
        tensor_in = torch.ones(1, 4, dtype=torch.int64).to(device)
        tensor_out = torch.zeros(world_size, 1, 4, dtype=torch.int64).to(device)
        mod = AllGatherInTensor()
        mod = mod.to(device)
        opt_mod = torch.compile(mod, dynamic=dynamic, fullgraph=True, backend=npu_backend)
        compile_result = opt_mod(tensor_out, tensor_in)
        given_shape = (world_size, 1, 4)
        results.append(given_shape == compile_result.shape)
        dist.destroy_process_group()  

    def _test_multiprocess(self, f, init_pg, world_size, dynamic):
        ctx = mp.get_context('spawn')
        ps = []
        manager = mp.Manager()
        results = manager.list()
        for rank in range(world_size):
            p = ctx.Process(target=f,
                            args=(rank, world_size, init_pg, dynamic, results))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
        return results

    def test_all_to_all_single_compile(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_split_size_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_split_size_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_fuse_add_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_fuse_add_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_all_to_all_single_compile_without_patch(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_split_size_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_split_size_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_fuse_add_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_fuse_add_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    @unittest.skipIf(True, "unsupported until cann support")
    def test_all_to_all_single_dims_compile(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_dims_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_dims_compile,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(True, "unsupported until cann support")
    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_all_to_all_single_dims_compile_without_patch(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_dims_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_dims_compile,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_all_to_all_single_export(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_export,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        shutil.rmtree("export_file")
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_export,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        shutil.rmtree("export_file")

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_all_to_all_single_export_without_patch(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_export,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        shutil.rmtree("export_file")
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_single_export,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        shutil.rmtree("export_file")

    def test_all_to_all(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_different_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_different_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_all_to_all_without_patch(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_different_size,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_to_all_different_size,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_allreduce(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_eager_allreduce_async,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_eager_allreduce_async,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allreduce,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allreduce,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_allreduce_without_patch(self):
        world_size = 4
        self.assertTrue(self._test_multiprocess(HcomTest._test_eager_allreduce_async,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_eager_allreduce_async,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allreduce,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allreduce,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_allgather(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_gather,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_gather,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_allgather_without_patch(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_gather,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_all_gather,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_allgather_into_tensor(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_into_tensor,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_into_tensor,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_allgather_into_tensor_without_patch(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_into_tensor,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_into_tensor,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_reduce_scatter_tensor(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_reduce_scatter_tensor,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_reduce_scatter_tensor,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_reduce_scatter_tensor_without_patch(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_reduce_scatter_tensor,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_reduce_scatter_tensor,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_broadcast(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_broadcast,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_broadcast,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_broadcast_without_patch(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_broadcast,
                                                HcomTest._init_dist_hccl_without_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_broadcast,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_allgather_in_tensor_uneven(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_uneven_same_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_uneven_different_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_uneven_same_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_uneven_different_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))

    def test_reducescatter_tensor_uneven(self):
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_reducescatter_tensor_uneven_same_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_reducescatter_tensor_uneven_different_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_reducescatter_tensor_uneven_same_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_reducescatter_tensor_uneven_different_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_uneven_without_patch(self):
        # 不支持动态
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_uneven_same_size,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_uneven_different_size,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_reducescatter_tensor_uneven_same_size,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_reducescatter_tensor_uneven_different_size,
                                                HcomTest._init_dist_hccl_without_patch, world_size, False))

    def test_patch_support_allgather_uneven(self):
        # 只支持打patch
        world_size = 2
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_reshape,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_reshape_1_4,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_reshape_8_1,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_no_same_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, False))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_reshape,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_reshape_1_4,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_reshape_8_1,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        self.assertTrue(self._test_multiprocess(HcomTest._test_allgather_in_tensor_no_same_size,
                                                HcomTest._init_dist_hccl_with_patch, world_size, True))
        
    def test_allgather_in_tensor_check_shape(self):
        world_size = 2
        results_dynamic_true = self._test_multiprocess(HcomTest._test_allgather_in_tensor_check_shape,
                                                       HcomTest._init_dist_hccl_with_patch, world_size, True)
        for result in results_dynamic_true:
            self.assertTrue(result)
        results_dynamic_false = self._test_multiprocess(HcomTest._test_allgather_in_tensor_check_shape,
                                                        HcomTest._init_dist_hccl_with_patch, world_size, False)
        for result in results_dynamic_false:
            self.assertTrue(result)

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_allgather_in_tensor_check_shape_without_patch(self):
        world_size = 2
        results_dynamic_true = self._test_multiprocess(HcomTest._test_allgather_in_tensor_check_shape,
                                                       HcomTest._init_dist_hccl_without_patch, world_size, True)
        for result in results_dynamic_true:
            self.assertTrue(result)
        results_dynamic_false = self._test_multiprocess(HcomTest._test_allgather_in_tensor_check_shape,
                                                        HcomTest._init_dist_hccl_without_patch, world_size, False)
        for result in results_dynamic_false:
            self.assertTrue(result)  


if __name__ == '__main__':
    unittest.main()
