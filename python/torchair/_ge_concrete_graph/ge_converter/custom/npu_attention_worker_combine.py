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
    Optional
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


@register_fx_node_ge_converter(torch.ops.npu.npu_attention_worker_combine.default)
def conveter_npu_attention_worker_combine_default(
        schedule_context: Tensor,
        expert_scales: Tensor,
        layer_id: Tensor,
        hidden_size: int,
        *,
        token_dtype: Optional[int] = 0,
        need_schedule: Optional[int] = 0,
        meta_outputs: List[TensorSpec] = None):
    return ge.AttentionWorkerCombine(schedule_context, expert_scales, layer_id,
                hidden_size=hidden_size, token_dtype=token_dtype, need_schedule=need_schedule)