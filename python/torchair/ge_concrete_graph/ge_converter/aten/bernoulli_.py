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


@register_fx_node_ge_converter(torch.ops.aten.bernoulli_.Tensor)
def conveter_aten_bernoulli__Tensor(
        self: Tensor,
        p: Tensor,
        *,
        generator: Optional[Generator] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bernoulli_.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bernoulli_.float)
def conveter_aten_bernoulli__float(
        self: Tensor,
        p: float = 0.5,
        *,
        generator: Optional[Generator] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bernoulli_.float ge converter is not implement!")


