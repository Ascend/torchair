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


@register_fx_node_ge_converter(torch.ops.aten.diagonal_scatter.default)
def conveter_aten_diagonal_scatter_default(
        self: Tensor,
        src: Tensor,
        offset: int = 0,
        dim1: int = 0,
        dim2: int = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.diagonal_scatter.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal_scatter.out)
def conveter_aten_diagonal_scatter_out(
        self: Tensor,
        src: Tensor,
        offset: int = 0,
        dim1: int = 0,
        dim2: int = 1,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::diagonal_scatter.out(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.diagonal_scatter.out ge converter is not implement!")


