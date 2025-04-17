from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Optional
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_init_routing_v2.default)
def conveter_npu_moe_init_routing_v2_default(
        x: Tensor,
        expert_idx: Tensor,
        *,
        scale: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        active_num: int = -1,
        expert_capacity: int = -1,
        expert_num: int = -1,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 0,
        expert_tokens_num_flag: bool = False,
        quant_mode: int = -1,
        active_expert_range: List[int] = [],
        row_idx_type: int = 0,
        meta_outputs: List[TensorSpec] = None,
):
    return ge.MoeInitRoutingV3(x, expert_idx, scale, offset,
                               active_num=active_num, expert_capacity=expert_capacity,
                               expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                               expert_tokens_num_type=expert_tokens_num_type,
                               expert_tokens_num_flag=expert_tokens_num_flag,
                               quant_mode=quant_mode, active_expert_range=active_expert_range,
                               row_idx_type=row_idx_type)
