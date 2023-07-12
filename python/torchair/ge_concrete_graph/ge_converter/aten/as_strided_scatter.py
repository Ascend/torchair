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


@register_fx_node_ge_converter(torch.ops.aten.as_strided_scatter.default)
def conveter_aten_as_strided_scatter_default(
        self: Tensor,
        src: Tensor,
        size: Union[List[int], Tensor],
        stride: Union[List[int], Tensor],
        storage_offset: Optional[Union[int, Tensor]] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.as_strided_scatter.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.as_strided_scatter.out)
def conveter_aten_as_strided_scatter_out(
        self: Tensor,
        src: Tensor,
        size: Union[List[int], Tensor],
        stride: Union[List[int], Tensor],
        storage_offset: Optional[Union[int, Tensor]] = None,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::as_strided_scatter.out(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.as_strided_scatter.out ge converter is not implement!")


