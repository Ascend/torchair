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


@register_fx_node_ge_converter(torch.ops.aten.clamp_min_.default)
def conveter_aten_clamp_min__default(
        self: Tensor,
        min: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp_min_.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min_.Tensor)
def conveter_aten_clamp_min__Tensor(
        self: Tensor,
        min: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clamp_min_.Tensor ge converter is not implement!")


