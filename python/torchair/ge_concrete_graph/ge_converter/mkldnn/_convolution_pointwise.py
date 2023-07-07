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


@register_fx_node_ge_converter(torch.ops.mkldnn._convolution_pointwise.default)
def conveter_mkldnn__convolution_pointwise_default(
        X: Tensor,
        W: Tensor,
        B: Optional[Tensor],
        padding: List[int],
        stride: List[int],
        dilation: List[int],
        groups: int,
        attr: str,
        scalars: Optional[Union[List[Number], Tensor]],
        algorithm: Optional[str],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: mkldnn::_convolution_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y """
    raise NotImplementedError("torch.ops.mkldnn._convolution_pointwise.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.mkldnn._convolution_pointwise.binary)
def conveter_mkldnn__convolution_pointwise_binary(
        X: Tensor,
        other: Tensor,
        W: Tensor,
        B: Optional[Tensor],
        padding: List[int],
        stride: List[int],
        dilation: List[int],
        groups: int,
        binary_attr: str,
        alpha: Optional[Union[Number, Tensor]],
        unary_attr: Optional[str],
        unary_scalars: Optional[Union[List[Number], Tensor]],
        unary_algorithm: Optional[str],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: mkldnn::_convolution_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor Y """
    raise NotImplementedError("torch.ops.mkldnn._convolution_pointwise.binary ge converter is not implement!")


