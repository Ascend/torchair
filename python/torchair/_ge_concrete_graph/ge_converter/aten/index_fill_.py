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


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.int_Tensor)
def conveter_aten_index_fill__int_Tensor(
    self: Tensor, dim: int, index: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.int_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.int_Scalar)
def conveter_aten_index_fill__int_Scalar(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.int_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.Dimname_Scalar)
def conveter_aten_index_fill__Dimname_Scalar(
    self: Tensor,
    dim: str,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_fill_.Dimname_Scalar(Tensor(a!) self, str dim, Tensor index, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.Dimname_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.Dimname_Tensor)
def conveter_aten_index_fill__Dimname_Tensor(
    self: Tensor, dim: str, index: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill_.Dimname_Tensor(Tensor(a!) self, str dim, Tensor index, Tensor value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.Dimname_Tensor ge_converter is not implemented!")
