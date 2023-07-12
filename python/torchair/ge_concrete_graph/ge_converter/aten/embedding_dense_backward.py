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


@register_fx_node_ge_converter(torch.ops.aten.embedding_dense_backward.default)
def conveter_aten_embedding_dense_backward_default(
        grad_output: Tensor,
        indices: Tensor,
        num_weights: Union[int, Tensor],
        padding_idx: Union[int, Tensor],
        scale_grad_by_freq: bool,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor """
    raise NotImplementedError("torch.ops.aten.embedding_dense_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.embedding_dense_backward.out)
def conveter_aten_embedding_dense_backward_out(
        grad_output: Tensor,
        indices: Tensor,
        num_weights: Union[int, Tensor],
        padding_idx: Union[int, Tensor],
        scale_grad_by_freq: bool,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::embedding_dense_backward.out(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.embedding_dense_backward.out ge converter is not implement!")


