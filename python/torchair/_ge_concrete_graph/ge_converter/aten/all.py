from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(BOOL(1, 2)),
    Support(BOOL(3, 4)),
    Support(F32(3, 4)),
    Support(F16(3, 4))
])
@register_fx_node_ge_converter(torch.ops.aten.all.default)
def conveter_aten_all_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::all(Tensor self) -> Tensor"""
    dim = list(range(self.rank))
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    return ge.ReduceAll(self, dim)


@register_fx_node_ge_converter(torch.ops.aten.all.dim)
def conveter_aten_all_dim(
    self: Tensor, dim: int, keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.all.dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.all.out)
def conveter_aten_all_out(
    self: Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.all.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.all.all_out)
def conveter_aten_all_all_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.all.all_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.all.dimname)
def conveter_aten_all_dimname(
    self: Tensor, dim: str, keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::all.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.all.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.all.dimname_out)
def conveter_aten_all_dimname_out(
    self: Tensor,
    dim: str,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::all.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.all.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.all.int)
def conveter_aten_all_int(self: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::all.int(int[] self) -> bool"""
    raise NotImplementedError("torch.ops.aten.all.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.all.float)
def conveter_aten_all_float(self: List[float], meta_outputs: TensorSpec = None):
    """NB: aten::all.float(float[] self) -> bool"""
    raise NotImplementedError("torch.ops.aten.all.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.all.bool)
def conveter_aten_all_bool(self: List[bool], meta_outputs: TensorSpec = None):
    """NB: aten::all.bool(bool[] self) -> bool"""
    raise NotImplementedError("torch.ops.aten.all.bool ge_converter is not implemented!")
