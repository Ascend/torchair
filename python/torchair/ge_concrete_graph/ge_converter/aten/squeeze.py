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


@register_fx_node_ge_converter(torch.ops.aten.squeeze.default)
def conveter_aten_squeeze_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::squeeze(Tensor(a) self) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.squeeze.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.squeeze.dim)
def conveter_aten_squeeze_dim(
        self: Tensor,
        dim: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.squeeze.dim ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.squeeze.dims)
def conveter_aten_squeeze_dims(
        self: Tensor,
        dim: List[int],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.squeeze.dims ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.squeeze.dimname)
def conveter_aten_squeeze_dimname(
        self: Tensor,
        dim: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::squeeze.dimname(Tensor(a) self, str dim) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.squeeze.dimname ge converter is not implement!")


