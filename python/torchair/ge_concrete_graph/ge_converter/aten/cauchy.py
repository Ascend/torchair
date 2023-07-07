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


@register_fx_node_ge_converter(torch.ops.aten.cauchy.default)
def conveter_aten_cauchy_default(
        self: Tensor,
        median: float = 0.0,
        sigma: float = 1.0,
        *,
        generator: Optional[Generator] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cauchy(Tensor self, float median=0., float sigma=1., *, Generator? generator=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.cauchy.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cauchy.out)
def conveter_aten_cauchy_out(
        self: Tensor,
        median: float = 0.0,
        sigma: float = 1.0,
        *,
        generator: Optional[Generator] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cauchy.out(Tensor self, float median=0., float sigma=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cauchy.out ge converter is not implement!")


