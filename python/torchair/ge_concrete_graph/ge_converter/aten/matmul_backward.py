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


@register_fx_node_ge_converter(torch.ops.aten.matmul_backward.default)
def conveter_aten_matmul_backward_default(
    grad: Tensor, 
    self: Tensor, 
    other: Tensor, 
    mask: List[bool], 
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::matmul_backward(Tensor grad, Tensor self, Tensor other, bool[2] mask) -> (Tensor, Tensor)"""
    return NotImplementedError("torch.ops.aten.matmul_backward.default ge_converter is not implemented!")

