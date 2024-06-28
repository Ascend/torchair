from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
from torch.library import Library
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import _world
from torchair.ge_concrete_graph.ge_graph import Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.utils import normalize_reduceop_type, get_group_name_and_record
from torchair.core.utils import logger

npu_define_lib = Library("npu_define", "DEF")
op_name = npu_define_lib.define(
    "allreduce(Tensor input, str reduce_type, int[] ranks, float? timeout=None) -> Tensor")


def allreduce_cpu(inputs, reduce_type, ranks, timeout=None):
    if custom_all_reduce is None:
        torch_all_reduce(inputs)
    else:
        custom_all_reduce(inputs)
    return inputs


def allreduce_npu(inputs, reduce_type, ranks, timeout=None):
    if custom_all_reduce is None:
        torch_all_reduce(inputs)
    else:
        custom_all_reduce(inputs)
    return inputs


def allreduce_meta(inputs, reduce_type, ranks, timeout=None):
    return inputs.new_empty(inputs.size())


npu_define_lib.impl(op_name, allreduce_cpu, 'CPU')
npu_define_lib.impl(op_name, allreduce_meta, 'Meta')
npu_define_lib.impl(op_name, allreduce_npu, 'PrivateUse1')


def npu_all_reduce(tensor, op="sum", group=None, async_op=False):
    rank_list = torch.distributed.get_process_group_ranks(_world.default_pg)
    tensor.copy_(torch.ops.npu_define.allreduce(tensor, "sum", rank_list))


torch_all_reduce = torch.distributed.all_reduce
torch.distributed.all_reduce = npu_all_reduce
custom_all_reduce = None


def get_npu_all_reduce():
    return npu_all_reduce


def backup_custom_all_reduce(func: None):
    if func is not None:
        custom_all_reduce = func


@register_fx_node_ge_converter(torch.ops.npu_define.allreduce.default)
def conveter_allreduce(
        self: Tensor,
        reduce_type,
        rank_list,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    group_name = get_group_name_and_record(c10d._world.pg_to_tag[_world.default_pg],
                                           rank_list, _world.default_pg.size())
    return ge.HcomAllReduce(self, reduction=normalize_reduceop_type(reduce_type), group=group_name, fusion=0)


def adapter_functional_collectives_all_reduce(tensor, op="sum", group=None, async_op=False):
    if not isinstance(op, str):
        raise TypeError("functional_collectives_context patch only sopport str reducetype")
    if group is None:
        tensor.copy_(torch.distributed._functional_collectives.all_reduce(tensor, op, _world.default_pg))
    else:
        tensor.copy_(torch.distributed._functional_collectives.all_reduce(tensor, op, group))


class functional_collectives_context:
    '''
    functional_collectives_context支持在作用域内将torch.distributed.all_reduce patch到
    torch.distributed._functional_collectives中的对应api中使用(当社区支持allreduce直接入图后, 该接口将被逐渐废弃)
    目的： 用户不用修改脚本,使用torch.distributed相关api也能torch.compile入图
    使用方式：
    with functional_collectives_context():
        opt_mod = torch.compile(mod, ...)
        compile_result = opt_mod(x)
    注意:
    1、该上下文管理器只能支持有限场景下的自动转化入图,如torch.distributed api不能使用如dist.ReduceOp.SUM这种数据类型,
    如使用上述相关入参将compile失败,此时用户需要手动修改脚本将torch.distributed api替换为 torch.distributed._functional_collectives中的api
    2、reduce_scatter_tensor并不需要在此patch,因为在原生torch中已经被处理过了
    '''
    def __init__(self) -> None:
        self.torch_all_reduce = None

    def __enter__(self):
        self.torch_all_reduce = torch.distributed.all_reduce
        torch.distributed.all_reduce = adapter_functional_collectives_all_reduce

    def __exit__(self, *args, **kwargs):
        torch.distributed.all_reduce = self.torch_all_reduce

