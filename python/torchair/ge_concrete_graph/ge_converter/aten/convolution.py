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


@register_fx_node_ge_converter(torch.ops.aten.convolution.default)
def conveter_aten_convolution_default(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: List[int],
    padding: Union[List[int], Tensor],
    dilation: List[int],
    transposed: bool,
    output_padding: Union[List[int], Tensor],
    groups: int,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.convolution.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.convolution.out)
def conveter_aten_convolution_out(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: List[int],
    padding: Union[List[int], Tensor],
    dilation: List[int],
    transposed: bool,
    output_padding: Union[List[int], Tensor],
    groups: int,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::convolution.out(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.convolution.out ge_converter is not implemented!")
