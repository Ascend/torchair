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


@register_fx_node_ge_converter(torch.ops.aten.im2col.default)
def conveter_aten_im2col_default(
        self: Tensor,
        kernel_size: List[int],
        dilation: List[int],
        padding: List[int],
        stride: List[int],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor """
    raise NotImplementedError("torch.ops.aten.im2col.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.im2col.out)
def conveter_aten_im2col_out(
        self: Tensor,
        kernel_size: List[int],
        dilation: List[int],
        padding: List[int],
        stride: List[int],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.im2col.out ge converter is not implement!")


