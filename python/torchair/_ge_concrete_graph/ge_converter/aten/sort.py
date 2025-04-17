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


@declare_supported(
    [
        Support(F32(1, 4, 2), 1),
        Support(F32(16, 4, 4, 1, 1), 1),
        Support(F32(16, 4, 4, 1, 1), 1, True),
        Support(I8(1, 4, 2), 1),
    ]
)

@register_fx_node_ge_converter(torch.ops.aten.sort.default)
def conveter_aten_sort_default(
    self: Tensor, dim: int = -1, descending: bool = False, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)"""
    if dim < 0:
        dim += self.rank
    last_dim = -1 + self.rank
    if dim != last_dim:
        perm = []
        for i in range(self.rank):
            perm.append(i)
        perm[dim], perm[last_dim] = perm[last_dim], perm[dim]
        perm = dtype_promote(perm, target_dtype=DataType.DT_INT64)
        transpose_self = ge.Transpose(self, perm)
        values, indices = ge.Sort(transpose_self, axis=last_dim, descending=descending)
        values = ge.Transpose(values, perm)
        indices = ge.Transpose(indices, perm)
    else:
        values, indices = ge.Sort(self, axis=last_dim, descending=descending)
    indices = dtype_promote(indices, target_dtype=DataType.DT_INT64)
    return values, indices


@register_fx_node_ge_converter(torch.ops.aten.sort.stable)
def conveter_aten_sort_stable(
    self: Tensor,
    *,
    stable: Optional[bool],
    dim: int = -1,
    descending: bool = False,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)"""
    raise RuntimeError("torch.ops.aten.sort.stable ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.values_stable)
def conveter_aten_sort_values_stable(
    self: Tensor,
    *,
    stable: Optional[bool],
    dim: int = -1,
    descending: bool = False,
    values: Tensor = None,
    indices: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise RuntimeError("torch.ops.aten.sort.values_stable ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.values)
def conveter_aten_sort_values(
    self: Tensor,
    dim: int = -1,
    descending: bool = False,
    *,
    values: Tensor = None,
    indices: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise RuntimeError("torch.ops.aten.sort.values ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.dimname)
def conveter_aten_sort_dimname(
    self: Tensor, dim: str, descending: bool = False, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort.dimname(Tensor self, str dim, bool descending=False) -> (Tensor values, Tensor indices)"""
    raise RuntimeError("torch.ops.aten.sort.dimname ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.dimname_values)
def conveter_aten_sort_dimname_values(
    self: Tensor,
    dim: str,
    descending: bool = False,
    *,
    values: Tensor = None,
    indices: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort.dimname_values(Tensor self, str dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise RuntimeError("torch.ops.aten.sort.dimname_values ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.dimname_stable)
def conveter_aten_sort_dimname_stable(
    self: Tensor,
    *,
    stable: Optional[bool],
    dim: str,
    descending: bool = False,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort.dimname_stable(Tensor self, *, bool? stable, str dim, bool descending=False) -> (Tensor values, Tensor indices)"""
    raise RuntimeError("torch.ops.aten.sort.dimname_stable ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.dimname_values_stable)
def conveter_aten_sort_dimname_values_stable(
    self: Tensor,
    *,
    stable: Optional[bool],
    dim: str,
    descending: bool = False,
    values: Tensor = None,
    indices: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::sort.dimname_values_stable(Tensor self, *, bool? stable, str dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise RuntimeError("torch.ops.aten.sort.dimname_values_stable ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.int)
def conveter_aten_sort_int(
    self: List[int], reverse: bool = False
):
    """NB: aten::sort.int(int[](a!) self, bool reverse=False) -> ()"""
    raise RuntimeError("torch.ops.aten.sort.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.float)
def conveter_aten_sort_float(
    self: List[float], reverse: bool = False
):
    """NB: aten::sort.float(float[](a!) self, bool reverse=False) -> ()"""
    raise RuntimeError("torch.ops.aten.sort.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.Tensor)
def conveter_aten_sort_Tensor(
    self: List[Tensor], reverse: bool = False
):
    """NB: aten::sort.Tensor(Tensor[](a!) self, bool reverse=False) -> ()"""
    raise RuntimeError("torch.ops.aten.sort.Tensor ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.bool)
def conveter_aten_sort_bool(
    self: List[bool], reverse: bool = False
):
    """NB: aten::sort.bool(bool[](a!) self, bool reverse=False) -> ()"""
    raise RuntimeError("torch.ops.aten.sort.bool ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sort.str)
def conveter_aten_sort_str(
    self: List[str], reverse: bool = False
):
    """NB: aten::sort.str(str[](a!) self, bool reverse=False) -> ()"""
    raise RuntimeError("torch.ops.aten.sort.str ge_converter is not supported!")
