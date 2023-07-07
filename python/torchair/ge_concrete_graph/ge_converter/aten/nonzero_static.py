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


@register_fx_node_ge_converter(torch.ops.aten.nonzero_static.default)
def conveter_aten_nonzero_static_default(
        self: Tensor,
        *,
        size: int,
        fill_value: int = -1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nonzero_static(Tensor self, *, int size, int fill_value=-1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.nonzero_static.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nonzero_static.out)
def conveter_aten_nonzero_static_out(
        self: Tensor,
        *,
        size: int,
        fill_value: int = -1,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nonzero_static.out(Tensor self, *, int size, int fill_value=-1, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.nonzero_static.out ge converter is not implement!")


