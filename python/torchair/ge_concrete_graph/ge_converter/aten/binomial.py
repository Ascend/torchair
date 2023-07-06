
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
@register_fx_node_ge_converter(torch.ops.aten.binomial.default)
def conveter_aten_binomial_default(
        count: Tensor,
        prob: Tensor,
        generator: Optional[Generator] = None,
        meta_outputs: Any = None):
    """ NB: aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.binomial.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.binomial.out)
def conveter_aten_binomial_out(
        count: Tensor,
        prob: Tensor,
        generator: Optional[Generator] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::binomial.out(Tensor count, Tensor prob, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.binomial.out ge converter is not implement!")


