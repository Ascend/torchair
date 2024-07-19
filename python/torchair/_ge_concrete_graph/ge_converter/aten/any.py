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

import sys
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported, DataType
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, F16, BOOL, Support


@declare_supported(
    [
        Support(BOOL(4, 2)),
        Support(F32(4, 2)),
        Support(F16(4, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.any.default)
def conveter_aten_any_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::any(Tensor self) -> Tensor"""
    dim = self.rank
    # ReduceAny only support input dtype of bool
    self = dtype_promote(self, target_dtype=DataType.DT_BOOL)

    if dim == 0:
        self = ge.Unsqueeze(self, axes=[0])
        dims = [0]
    else:
        dims = [i for i in range(dim)]
        
    return ge.ReduceAny(self, axes=dims, keep_dims=False)


@declare_supported(
    [
        Support(F32(4, 2), 0),
        Support(F32(4, 2), 1),
        Support(BOOL(4, 2), 0, keepdim=True),
        Support(BOOL(4, 2), 1, keepdim=True),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.any.dim)
def conveter_aten_any_dim(
    self: Tensor, dim: int, keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""
    # ReduceAny only support input dtype of bool
    self = dtype_promote(self, target_dtype=DataType.DT_BOOL)

    if dim == -sys.maxsize - 1:
        dims = [i for i in range(dim)]
    else:
        dims = [dim]

    return ge.ReduceAny(self, axes=dims, keep_dims=keepdim)


@register_fx_node_ge_converter(torch.ops.aten.any.out)
def conveter_aten_any_out(
    self: Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.any.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.any.all_out)
def conveter_aten_any_all_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.any.all_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.any.dimname)
def conveter_aten_any_dimname(
    self: Tensor, dim: str, keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::any.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.any.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.any.dimname_out)
def conveter_aten_any_dimname_out(
    self: Tensor,
    dim: str,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::any.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.any.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.any.str)
def conveter_aten_any_str(self: List[str], meta_outputs: TensorSpec = None):
    """NB: aten::any.str(str[] self) -> bool"""
    raise NotImplementedError("torch.ops.aten.any.str ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.any.int)
def conveter_aten_any_int(self: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::any.int(int[] self) -> bool"""
    raise NotImplementedError("torch.ops.aten.any.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.any.float)
def conveter_aten_any_float(self: List[float], meta_outputs: TensorSpec = None):
    """NB: aten::any.float(float[] self) -> bool"""
    raise NotImplementedError("torch.ops.aten.any.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.any.bool)
def conveter_aten_any_bool(self: List[bool], meta_outputs: TensorSpec = None):
    """NB: aten::any.bool(bool[] self) -> bool"""
    raise NotImplementedError("torch.ops.aten.any.bool ge_converter is not implemented!")
