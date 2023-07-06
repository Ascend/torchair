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


@register_fx_node_ge_converter(torch.ops.aten.native_dropout_backward.default)
def conveter_aten_native_dropout_backward_default(
        grad_output: Tensor,
        mask: Tensor,
        scale: float,
        meta_outputs: Any = None):
    """ NB: aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor """
    raise NotImplementedError("torch.ops.aten.native_dropout_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.native_dropout_backward.out)
def conveter_aten_native_dropout_backward_out(
        grad_output: Tensor,
        mask: Tensor,
        scale: float,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::native_dropout_backward.out(Tensor grad_output, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.native_dropout_backward.out ge converter is not implement!")


