import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
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


@register_fx_node_ge_converter(torch.ops.aten.cummin.default)
def conveter_aten_cummin_default(
        self: Tensor,
        dim: int,
        meta_outputs: Any = None):
    """ NB: aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.cummin.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cummin.dimname)
def conveter_aten_cummin_dimname(
        self: Tensor,
        dim: str,
        meta_outputs: Any = None):
    """ NB: aten::cummin.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices) """
    raise NotImplementedError("torch.ops.aten.cummin.dimname ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cummin.dimname_out)
def conveter_aten_cummin_dimname_out(
        self: Tensor,
        dim: str,
        *,
        values: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::cummin.dimname_out(Tensor self, str dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.cummin.dimname_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cummin.out)
def conveter_aten_cummin_out(
        self: Tensor,
        dim: int,
        *,
        values: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices) """
    raise NotImplementedError("torch.ops.aten.cummin.out ge converter is not implement!")


