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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.col2im.default)
def conveter_aten_col2im_default(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.col2im.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.col2im.out)
def conveter_aten_col2im_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::col2im.out(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.col2im.out ge_converter is not implemented!")
