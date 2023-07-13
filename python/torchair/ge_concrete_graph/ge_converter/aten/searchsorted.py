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


@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Tensor)
def conveter_aten_searchsorted_Tensor(
        sorted_sequence: Tensor,
        self: Tensor,
        *,
        out_int32: bool = False,
        right: bool = False,
        side: Optional[str] = None,
        sorter: Optional[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.searchsorted.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Tensor_out)
def conveter_aten_searchsorted_Tensor_out(
        sorted_sequence: Tensor,
        self: Tensor,
        *,
        out_int32: bool = False,
        right: bool = False,
        side: Optional[str] = None,
        sorter: Optional[Tensor] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.searchsorted.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Scalar)
def conveter_aten_searchsorted_Scalar(
        sorted_sequence: Tensor,
        self: Union[Number, Tensor],
        *,
        out_int32: bool = False,
        right: bool = False,
        side: Optional[str] = None,
        sorter: Optional[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.searchsorted.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Scalar_out)
def conveter_aten_searchsorted_Scalar_out(
        sorted_sequence: Tensor,
        self: Union[Number, Tensor],
        *,
        out_int32: bool = False,
        right: bool = False,
        side: Optional[str] = None,
        sorter: Optional[Tensor] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.searchsorted.Scalar_out ge converter is not implement!")


