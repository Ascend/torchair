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


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Tensor)
def conveter_aten_bucketize_Tensor(
    self: Tensor,
    boundaries: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bucketize.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Scalar)
def conveter_aten_bucketize_Scalar(
    self: Union[Number, Tensor],
    boundaries: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bucketize.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Tensor_out)
def conveter_aten_bucketize_Tensor_out(
    self: Tensor,
    boundaries: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bucketize.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Scalar_out)
def conveter_aten_bucketize_Scalar_out(
    self: Union[Number, Tensor],
    boundaries: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bucketize.Scalar_out(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bucketize.Scalar_out ge_converter is not implemented!")
