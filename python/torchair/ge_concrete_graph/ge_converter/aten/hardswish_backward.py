from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.hardswish_backward.default)
def conveter_aten_hardswish_backward_default(
    grad_output: Tensor, self: Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.hardswish_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.hardswish_backward.out)
def conveter_aten_hardswish_backward_out(
    grad_output: Tensor, self: Tensor, *, out: Tensor = None, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::hardswish_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.hardswish_backward.out ge_converter is not implemented!")
