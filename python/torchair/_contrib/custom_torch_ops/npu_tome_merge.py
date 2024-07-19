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
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F16(64, 16384, 1280), F16(64, 16384, 1280), I64(1, 16384), I64(1, 16384)),
])
@register_fx_node_ge_converter(torch.ops.npu_inference.npu_tome_merge.default)
def conveter_npu_tome_merge_default(
    token_a: Tensor,
    token_b: Tensor,
    token_indice: Tensor,
    arg_max: Tensor,
    top_rate: float = 0.5,
    meta_outputs: List[TensorSpec] = None
):
    return ge.TomeMerge(token_a, token_b, token_indice, arg_max, top_rate)