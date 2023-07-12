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


@register_fx_node_ge_converter(torch.ops.aten.log_normal.default)
def conveter_aten_log_normal_default(
        self: Tensor,
        mean: float = 1.0,
        std: float = 2.0,
        *,
        generator: Optional[Generator] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log_normal(Tensor self, float mean=1., float std=2., *, Generator? generator=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.log_normal.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log_normal.out)
def conveter_aten_log_normal_out(
        self: Tensor,
        mean: float = 1.0,
        std: float = 2.0,
        *,
        generator: Optional[Generator] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log_normal.out(Tensor self, float mean=1., float std=2., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.log_normal.out ge converter is not implement!")


