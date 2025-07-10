from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_distribute_dispatch_v2.default)
def convert_npu_moe_distribute_dispatch_v2(
    x: Tensor,
    expert_ids: Tensor,
    group_ep: str,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    *,
    scales: Optional[Tensor] = None,
    x_active_mask: Optional[Tensor] = None,
    expert_scales: Optional[Tensor] = None,
    group_tp: str = "",
    tp_world_size: int = 0,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    global_bs: int = 0,
    expert_token_nums_type: int = 1,
    meta_outputs: TensorSpec = None
):

    return ge.MoeDistributeDispatchV2(x,
                                   expert_ids,
                                   scales=scales,
                                   x_active_mask=x_active_mask,
                                   expert_scales=expert_scales,
                                   group_ep=group_ep,
                                   ep_world_size=ep_world_size,
                                   ep_rank_id=ep_rank_id,
                                   moe_expert_num=moe_expert_num,
                                   group_tp=group_tp,
                                   tp_world_size=tp_world_size,
                                   tp_rank_id=tp_rank_id,
                                   expert_shard_type=expert_shard_type,
                                   shared_expert_num=shared_expert_num,
                                   shared_expert_rank_num=shared_expert_rank_num,
                                   quant_mode=quant_mode,
                                   global_bs=global_bs,
                                   expert_token_nums_type=expert_token_nums_type)