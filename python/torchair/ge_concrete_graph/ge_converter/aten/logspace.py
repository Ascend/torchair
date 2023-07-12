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


@register_fx_node_ge_converter(torch.ops.aten.logspace.default)
def conveter_aten_logspace_default(
        start: Union[Number, Tensor],
        end: Union[Number, Tensor],
        steps: int,
        base: float = 10.0,
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::logspace(Scalar start, Scalar end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.logspace.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.logspace.out)
def conveter_aten_logspace_out(
        start: Union[Number, Tensor],
        end: Union[Number, Tensor],
        steps: int,
        base: float = 10.0,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::logspace.out(Scalar start, Scalar end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.logspace.out ge converter is not implement!")


