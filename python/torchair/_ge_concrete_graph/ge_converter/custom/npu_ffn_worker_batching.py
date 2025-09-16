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


@register_fx_node_ge_converter(torch.ops.npu.npu_ffn_worker_batching.default)
def convert_npu_ffn_worker_batching_default(
    schedule_context: Tensor,
    expert_num: int,
    max_out_shape: List[int],
    *,
    token_dtype: int = 0,
    need_schedule: int = 0,
    layer_num: int = 0,
    meta_outputs: List[TensorSpec] = None
):
    return ge.FfnWorkerBatching(schedule_context, expert_num=expert_num, max_out_shape=max_out_shape,
                                token_dtype=token_dtype, need_schedule=need_schedule, layer_num=layer_num)
