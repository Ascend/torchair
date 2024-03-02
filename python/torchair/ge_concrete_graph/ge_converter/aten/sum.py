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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported([
    Support(F32(3, 8, 4), dim=[1], keepdim=True),
    Support(F32(3, 8, 4), dim=[2], keepdim=False),
])
@register_fx_node_ge_converter(torch.ops.aten.sum.dim_IntList)
def conveter_aten_sum_dim_IntList(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    if len(dim) == 0:
        dim = list(range(self.rank))
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    return ge.ReduceSum(self, dim, keep_dims=keepdim)


@declare_supported(
    [
        Support(F32(2, 4)),
        Support(F32(2, 6, 12, 12)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.sum.default)
def conveter_aten_sum_default(
    self: Tensor, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    dimlist = [i for i in range(self.rank)]
    return ge.ReduceSum(ge.Cast(self, dst_type=meta_outputs.dtype), dimlist)


@register_fx_node_ge_converter(torch.ops.aten.sum.dim_DimnameList)
def conveter_aten_sum_dim_DimnameList(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sum.dim_DimnameList(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.sum.dim_DimnameList is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.sum.DimnameList_out)
def conveter_aten_sum_DimnameList_out(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sum.DimnameList_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sum.DimnameList_out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.sum.IntList_out)
def conveter_aten_sum_IntList_out(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sum.IntList_out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.sum.out)
def conveter_aten_sum_out(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sum.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sum.out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.sum.int)
def conveter_aten_sum_int(self: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::sum.int(int[] self) -> int"""
    raise RuntimeError("torch.ops.aten.sum.int is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.sum.float)
def conveter_aten_sum_float(self: List[float], meta_outputs: TensorSpec = None):
    """NB: aten::sum.float(float[] self) -> float"""
    raise RuntimeError("torch.ops.aten.sum.float is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.sum.complex)
def conveter_aten_sum_complex(self: List[complex], meta_outputs: TensorSpec = None):
    """NB: aten::sum.complex(complex[] self) -> complex"""
    raise RuntimeError("torch.ops.aten.sum.complex is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.sum.bool)
def conveter_aten_sum_bool(self: List[bool], meta_outputs: TensorSpec = None):
    """NB: aten::sum.bool(bool[] self) -> int"""
    raise RuntimeError("torch.ops.aten.sum.bool is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")
