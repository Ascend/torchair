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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(2, 2, 5), 1),
    Support(F32(2, 2, 5), 0),
    Support(F16(2, 2, 5), 1, dtype=torch.float32),
])
@register_fx_node_ge_converter(torch.ops.aten.cumsum.default)
def conveter_aten_cumsum_default(
    self: Tensor, dim: int, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""
    self = dtype_promote(self, target_dtype=dtype) if dtype else self
    return ge.Cumsum(self, dim)


@register_fx_node_ge_converter(torch.ops.aten.cumsum.dimname)
def conveter_aten_cumsum_dimname(
    self: Tensor, dim: str, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumsum.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cumsum.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumsum.dimname_out)
def conveter_aten_cumsum_dimname_out(
    self: Tensor,
    dim: str,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cumsum.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumsum.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumsum.out)
def conveter_aten_cumsum_out(
    self: Tensor,
    dim: int,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumsum.out ge_converter is not implemented!")
