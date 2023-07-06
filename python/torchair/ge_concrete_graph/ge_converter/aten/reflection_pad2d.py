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


@register_fx_node_ge_converter(torch.ops.aten.reflection_pad2d.default)
def conveter_aten_reflection_pad2d_default(
        self: Tensor,
        padding: Union[List[int], Tensor],
        meta_outputs: Any = None):
    """ NB: aten::reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor """
    raise NotImplementedError("torch.ops.aten.reflection_pad2d.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.reflection_pad2d.out)
def conveter_aten_reflection_pad2d_out(
        self: Tensor,
        padding: Union[List[int], Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::reflection_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.reflection_pad2d.out ge converter is not implement!")


