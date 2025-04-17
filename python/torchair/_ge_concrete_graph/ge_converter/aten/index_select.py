from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(3, 4), 1, I32(2, value_range=(0, 4))),
    Support(F32(3, 4), -1, I32(2, value_range=(0, 4))),
    Support(F32(3, 10), 1, I32(2, value_range=(0, 10))),
])
@register_fx_node_ge_converter(torch.ops.aten.index_select.default)
def conveter_aten_index_select_default(
    self: Tensor, dim: int, index: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_select(Tensor self, int dim, Tensor index) -> Tensor"""
    dim = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    return ge.GatherV2(self, index, dim)


@register_fx_node_ge_converter(torch.ops.aten.index_select.out)
def conveter_aten_index_select_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.index_select.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.index_select.dimname)
def conveter_aten_index_select_dimname(
    self: Tensor, dim: str, index: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_select.dimname(Tensor self, str dim, Tensor index) -> Tensor"""
    raise RuntimeError("torch.ops.aten.index_select.dimname is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.index_select.dimname_out)
def conveter_aten_index_select_dimname_out(
    self: Tensor,
    dim: str,
    index: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_select.dimname_out(Tensor self, str dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.index_select.dimname_out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
