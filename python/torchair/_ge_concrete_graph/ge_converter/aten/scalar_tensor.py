from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.scalar_tensor.default)
def conveter_aten_scalar_tensor_default(
    s: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is not None:
        # When a dtype is specified, ensure that the input scalar can be converted to the required dtype by inserting
        # a ge.Cast or ge.Const node if necessary.
        return dtype_promote(s, target_dtype=dtype)
    if isinstance(s, Tensor):
        return s
    return ge.Const(s)


@register_fx_node_ge_converter(torch.ops.aten.scalar_tensor.out)
def conveter_aten_scalar_tensor_out(
    s: Union[Number, Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::scalar_tensor.out(Scalar s, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scalar_tensor.out ge_converter is not implemented!")
