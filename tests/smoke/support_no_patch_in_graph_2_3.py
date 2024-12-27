import os
import torch
import torch.distributed as dist
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)


# 本文件用例看护接入原生通信算子convert的实现，根据迭代进展逐步添加。与其他几个用例的不同之处在于不会默认打patch
# 后续当支持全部算子都不打patch时，需要将不同smoke中的用例合并
if torch.__version__ < '2.3.1':
    torchair.patch_for_hcom()


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
        torch.distributed.all_gather_into_tensor(out_tensor, x, dist.group.WORLD)
        return out_tensor


class AllGatherInTensorNoGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x):
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
            torch.distributed.reduce_scatter_tensor(out_tensor, input_tensor, group=dist.group.WORLD)
        return out_tensor


def test_allreduce(rank, world_size):
    torch.npu.set_device(rank)
    torch.distributed.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones([2, 2], dtype=torch.int32).to("npu:" + str(rank))
    y = torch.ones([2, 2], dtype=torch.int32).to("npu:" + str(rank))
    model = torch.compile(AllReduce().to("npu:" + str(rank)), backend=npu_backend, dynamic=False)
    out = torch.ones([2, 2], dtype=torch.int32).npu() * 2 * world_size
    ret = model(x, y)
    assert out.equal(ret)
    torch.distributed.destroy_process_group()


def test_allgather_dynamic(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = [torch.zeros(2, 2, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
    mod = AllGather()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=True, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    for i, t in enumerate(compile_result):
        assert t.equal(ori_result[i])


def test_allgather_static(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = [torch.zeros(2, 2, dtype=torch.int64).to("npu:" + str(rank)) for _ in range(2)]
    mod = AllGather()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    for i, t in enumerate(compile_result):
        assert t.equal(ori_result[i])


def test_allgather_in_tensor_dynamic(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = torch.zeros(4, 2, dtype=torch.int64).to("npu:" + str(rank))
    mod = AllGatherInTensor()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=True, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    assert ori_result.equal(compile_result)


def test_allgather_in_tensor_static(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = torch.zeros(4, 2, dtype=torch.int64).to("npu:" + str(rank))
    mod = AllGatherInTensor()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    assert ori_result.equal(compile_result)


def test_reduce_scatter_tensor_static(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    input_tensor = torch.arange(world_size * 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    output_tensor = torch.zeros(2, dtype=torch.int64).to("npu:" + str(rank))
    mod = ReduceScatterTensor()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(output_tensor, input_tensor)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(output_tensor, input_tensor)
    assert ori_result.equal(compile_result)


def test_allgather_in_tensor_no_group(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("npu:" + str(rank)) + 1 + 2 * rank
    tensor_list = torch.zeros(4, 2, dtype=torch.int64).to("npu:" + str(rank))
    mod = AllGatherInTensorNoGroup()
    mod = mod.to("npu:" + str(rank))
    ori_result = mod(tensor_list, x)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True, backend=npu_backend)
    compile_result = opt_mod(tensor_list, x)
    assert ori_result.equal(compile_result)


def mp():
    world_size = 2
    # == == == == == == == == =  case1 allgather动态入图 == == == == == == == == ==
    torch.multiprocessing.spawn(test_allgather_dynamic, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 1 pass =============================", flush=True)
    # =================  case 2 allgather静态入图==================
    torch.multiprocessing.spawn(test_allgather_static, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 2 pass =============================", flush=True)
    # =================  case 3 allgather_in_tensor动态入图==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_dynamic, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 3 pass =============================", flush=True)
    # =================  case 4 allgather_in_tensor静态入图==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_static, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 4 pass =============================", flush=True)
    # =================  case 5 allreduce静态入图==================
    torch.multiprocessing.spawn(test_allreduce, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 5 pass =============================", flush=True)
    # =================  case 6 reduce_scatter_tensor静态入图==================
    torch.multiprocessing.spawn(test_reduce_scatter_tensor_static, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 6 pass =============================", flush=True)
    # =================  case 7 修复社区不能不带group入参入图的bug==================
    torch.multiprocessing.spawn(test_allgather_in_tensor_no_group, args=(world_size,), nprocs=world_size, join=True)
    print("==================case 7 pass =============================", flush=True)


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"
    mp()
