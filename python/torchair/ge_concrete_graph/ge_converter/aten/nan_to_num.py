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


@register_fx_node_ge_converter(torch.ops.aten.nan_to_num.default)
def conveter_aten_nan_to_num_default(
        self: Tensor,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        meta_outputs: Any = None):
    """ NB: aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.nan_to_num.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nan_to_num.out)
def conveter_aten_nan_to_num_out(
        self: Tensor,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.nan_to_num.out ge converter is not implement!")


