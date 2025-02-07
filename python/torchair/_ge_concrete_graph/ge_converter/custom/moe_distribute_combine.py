from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support

@register_fx_node_ge_converter(torch.ops.npu.npu_moe_distribute_combine.default)
def convert_npu_moe_distribute_combine(
    expand_x: Tensor,
    expert_ids: Tensor,
    expand_idx: Tensor,
    ep_send_counts: Tensor,
    tp_send_counts: Tensor,
    expert_scales: Tensor,
    group_ep: str,
    group_tp: str,
    ep_world_size: int,
    tp_world_size: int,
    ep_rank_id: int,
    tp_rank_id: int,
    expert_shard_type: int,
    shared_expert_rank_num: int,
    moe_expert_num: int,
    *,
    global_bs: int = 0,
    meta_outputs: TensorSpec = None
):

    return ge.MoeDistributeCombine(expand_x=expand_x,
                                   expert_ids=expert_ids,
                                   expand_idx=expand_idx,
                                   ep_send_counts=ep_send_counts,
                                   tp_send_counts=tp_send_counts,
                                   expert_scales=expert_scales,
                                   group_ep=group_ep,
                                   group_tp=group_tp,
                                   ep_world_size=ep_world_size,
                                   tp_world_size=tp_world_size,
                                   ep_rank_id=ep_rank_id,
                                   tp_rank_id=tp_rank_id,
                                   expert_shard_type=expert_shard_type,
                                   shared_expert_rank_num=shared_expert_rank_num,
                                   moe_expert_num=moe_expert_num,
                                   global_bs=global_bs)