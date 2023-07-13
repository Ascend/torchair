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


@register_fx_node_ge_converter(torch.ops.aten.max_pool2d_with_indices.default)
def conveter_aten_max_pool2d_with_indices_default(
        self: Tensor,
        kernel_size: List[int],
        stride: List[int] = [],
        padding: List[int] = [0, 0],
        dilation: List[int] = [1, 1],
        ceil_mode: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten.max_pool2d_with_indices.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.max_pool2d_with_indices.out)
def conveter_aten_max_pool2d_with_indices_out(
        self: Tensor,
        kernel_size: List[int],
        stride: List[int] = [],
        padding: List[int] = [0, 0],
        dilation: List[int] = [1, 1],
        ceil_mode: bool = False,
        *,
        out: Tensor = None,
        indices: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!)) """
    raise NotImplementedError("torch.ops.aten.max_pool2d_with_indices.out ge converter is not implement!")


