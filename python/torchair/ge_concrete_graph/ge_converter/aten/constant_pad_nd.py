import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
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


@register_fx_node_ge_converter(torch.ops.aten.constant_pad_nd.default)
def conveter_aten_constant_pad_nd_default(
        self: Tensor,
        pad: Union[List[int], Tensor],
        value: Union[Number, Tensor] = 0,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor """
    raise NotImplementedError("torch.ops.aten.constant_pad_nd.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.constant_pad_nd.out)
def conveter_aten_constant_pad_nd_out(
        self: Tensor,
        pad: Union[List[int], Tensor],
        value: Union[Number, Tensor] = 0,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::constant_pad_nd.out(Tensor self, SymInt[] pad, Scalar value=0, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.constant_pad_nd.out ge converter is not implement!")


