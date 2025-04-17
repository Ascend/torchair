from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(2, 2), dim=0, half_to_float=False),
        Support(F32(2, 2), dim=1, half_to_float=True),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._log_softmax.default)
def conveter_aten__log_softmax_default(
    self: Tensor, dim: int, half_to_float: bool, meta_outputs: TensorSpec = None
):
    """NB: aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    output = ge.LogSoftmaxV2(self, axes=[dim])
    return output


@register_fx_node_ge_converter(torch.ops.aten._log_softmax.out)
def conveter_aten__log_softmax_out(
    self: Tensor,
    dim: int,
    half_to_float: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten._log_softmax.out ge_converter is not supported!")
