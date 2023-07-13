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


@register_fx_node_ge_converter(torch.ops.aten.diagonal_backward.default)
def conveter_aten_diagonal_backward_default(
        grad_output: Tensor,
        input_sizes: Union[List[int], Tensor],
        offset: int,
        dim1: int,
        dim2: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor """
    raise NotImplementedError("torch.ops.aten.diagonal_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal_backward.out)
def conveter_aten_diagonal_backward_out(
        grad_output: Tensor,
        input_sizes: Union[List[int], Tensor],
        offset: int,
        dim1: int,
        dim2: int,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::diagonal_backward.out(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.diagonal_backward.out ge converter is not implement!")


