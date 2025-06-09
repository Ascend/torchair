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


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_distribute_combine_add_rms_norm.default)
def convert_npu_moe_distribute_combine_add_rms_norm(
    expand_x: Tensor,
    expert_ids: Tensor,
    expand_idx: Tensor,
    ep_send_counts: Tensor,
    expert_scales: Tensor,
    residual_x: Tensor,
    gamma: Tensor,
    group_ep: str,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    *,
    tp_send_counts: Tensor = None,
    x_active_mask: Tensor = None,
    activation_scale: Tensor = None,
    weight_scale: Tensor = None,
    group_list: Tensor = None,
    expand_scales: Tensor = None,
    shared_expert_x: Tensor = None,
    group_tp: str = "",
    tp_world_size: int = 0,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    global_bs: int = 0,
    out_dtype: int = 0,
    comm_quant_mode: int = 0,
    group_list_type: int = 0,
    comm_alg: str = "",
    norm_eps: float = 1e-6,
    meta_outputs: TensorSpec = None
):
    return ge.MoeDistributeCombineAddRmsNorm(expand_x=expand_x, 
                                   expert_ids=expert_ids, 
                                   assist_info=expand_idx, 
                                   ep_send_counts=ep_send_counts, 
                                   expert_scales=expert_scales, 
                                   residual_x=residual_x, 
                                   gamma=gamma, 
                                   tp_send_counts=tp_send_counts, 
                                   x_active_mask=x_active_mask, 
                                   activation_scale=activation_scale, 
                                   weight_scale=weight_scale, 
                                   group_list=group_list, 
                                   expand_scales=expand_scales, 
                                   shared_expert_x=shared_expert_x, 
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
                                   global_bs=global_bs, 
                                   out_dtype=out_dtype, 
                                   comm_quant_mode=comm_quant_mode, 
                                   group_list_type=group_list_type, 
                                   comm_alg=comm_alg, 
                                   norm_eps=norm_eps)