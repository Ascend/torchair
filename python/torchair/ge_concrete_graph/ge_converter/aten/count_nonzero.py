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


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.dim_IntList)
def conveter_aten_count_nonzero_dim_IntList(
        self: Tensor,
        dim: List[int],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor """
    raise NotImplementedError("torch.ops.aten.count_nonzero.dim_IntList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.dim_IntList_out)
def conveter_aten_count_nonzero_dim_IntList_out(
        self: Tensor,
        dim: List[int],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::count_nonzero.dim_IntList_out(Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.count_nonzero.dim_IntList_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.default)
def conveter_aten_count_nonzero_default(
        self: Tensor,
        dim: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::count_nonzero(Tensor self, int? dim=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.count_nonzero.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.out)
def conveter_aten_count_nonzero_out(
        self: Tensor,
        dim: Optional[int] = None,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::count_nonzero.out(Tensor self, int? dim=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.count_nonzero.out ge converter is not implement!")


