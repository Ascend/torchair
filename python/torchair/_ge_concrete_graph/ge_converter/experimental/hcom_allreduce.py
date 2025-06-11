import torch
from torch.distributed.distributed_c10d import _world
from torch.library import Library

from torchair.core.utils import logger


npu_define_lib = Library("npu_define", "DEF")


def convert_reduce_op(op):
    if isinstance(op, str):
        return op
    # 无法使用map类型的表驱动，因为torch2.1版本中symbolic_convert中的BUILD_MAP无法处理C++枚举类型的ReduceOp
    if op is torch.distributed.ReduceOp.SUM:
        return "sum"
    elif op is torch.distributed.ReduceOp.AVG:
        return "avg"
    elif op is torch.distributed.ReduceOp.PRODUCT:
        return "product"
    elif op is torch.distributed.ReduceOp.MIN:
        return "min"
    elif op is torch.distributed.ReduceOp.MAX:
        return "max"
    elif op is torch.distributed.ReduceOp.BAND:
        return "bor"
    elif op is torch.distributed.ReduceOp.BXOR:
        return "bxor"
    else:
        raise ValueError(f"Unsupported reduce op: {op}")


def npu_allreduce_patch_dist(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    if not torch.distributed._functional_collectives._are_we_tracing():
        return torch.distributed.distributed_c10d.all_reduce(tensor, op, group, async_op)
    if async_op:
        raise AssertionError(f'When you enable torch.compile or use the cache_compile feature, '
                       f'use the patch_for_hcom interface to ensure that collective communication functions '
                       f'are included in the graph. However, unlike the eager mode, the compile mode '
                       f'does not support the async_op = True parameter for collective communication APIs.')
    op = convert_reduce_op(op)
    group = _world.default_pg if group is None else group
    tensor.copy_(torch.distributed._functional_collectives.all_reduce(tensor, op, group))


def patch_for_deepspeed_allreduce():
    try:
        from deepspeed import comm as dist
    except Exception as e:
        logger.info(f'env import deepspeed error {str(e)}, only patch pytorch dist api')
    else:
        dist.all_reduce = npu_allreduce_patch_dist
        # Adapt deepspeed version later than v0.10.0,
        # inference_all_reduce and all_reduce have save torch backend implementation.
        dist.inference_all_reduce = npu_allreduce_patch_dist
