from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(0.125, 0.875, 7, dtype=torch.float32),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.linspace.default)
def conveter_aten_linspace_default(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if steps < 0:
        raise AssertionError("number of steps must be non-negative")
    if layout is not None and layout != torch.strided:
        raise RuntimeError("torch.ops.aten.linspace.default ge_converter is only supported on dense tensor now!")
    result = ge.LinSpace(start, end, steps)
    if dtype is not None:
        result = dtype_promote(result, target_dtype=dtype)
    return result


@register_fx_node_ge_converter(torch.ops.aten.linspace.out)
def conveter_aten_linspace_out(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.linspace.out ge_converter is not supported!")
