import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Tensor)
def conveter_aten_bucketize_Tensor(
        self: Tensor,
        boundaries: Tensor,
        *,
        out_int32: bool = False,
        right: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bucketize.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Scalar)
def conveter_aten_bucketize_Scalar(
        self: Union[Number, Tensor],
        boundaries: Tensor,
        *,
        out_int32: bool = False,
        right: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bucketize.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Tensor_out)
def conveter_aten_bucketize_Tensor_out(
        self: Tensor,
        boundaries: Tensor,
        *,
        out_int32: bool = False,
        right: bool = False,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bucketize.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bucketize.Scalar_out)
def conveter_aten_bucketize_Scalar_out(
        self: Union[Number, Tensor],
        boundaries: Tensor,
        *,
        out_int32: bool = False,
        right: bool = False,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bucketize.Scalar_out(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bucketize.Scalar_out ge converter is not implement!")


