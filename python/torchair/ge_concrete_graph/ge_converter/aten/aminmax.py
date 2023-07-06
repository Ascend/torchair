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


@register_fx_node_ge_converter(torch.ops.aten.aminmax.default)
def conveter_aten_aminmax_default(
        self: Tensor,
        *,
        dim: Optional[int] = None,
        keepdim: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max) """
    raise NotImplementedError("torch.ops.aten.aminmax.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.aminmax.out)
def conveter_aten_aminmax_out(
        self: Tensor,
        *,
        dim: Optional[int] = None,
        keepdim: bool = False,
        min: Tensor = None,
        max: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max) """
    raise NotImplementedError("torch.ops.aten.aminmax.out ge converter is not implement!")


