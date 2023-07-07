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


@register_fx_node_ge_converter(torch.ops.aten.cummax.default)
def conveter_aten_cummax_default(
        self: Tensor,
        dim: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.cummax.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cummax.dimname)
def conveter_aten_cummax_dimname(
        self: Tensor,
        dim: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cummax.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.cummax.dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cummax.dimname_out)
def conveter_aten_cummax_dimname_out(
        self: Tensor,
        dim: str,
        *,
        values: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cummax.dimname_out(Tensor self, str dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.cummax.dimname_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cummax.out)
def conveter_aten_cummax_out(
        self: Tensor,
        dim: int,
        *,
        values: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.cummax.out ge converter is not implement!")


