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


@register_fx_node_ge_converter(torch.ops.aten._foreach_neg.default)
def conveter_aten__foreach_neg_default(
        self: List[Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_neg(Tensor[] self) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_neg.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_neg.out)
def conveter_aten__foreach_neg_out(
        self: List[Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_neg.out(Tensor[] self, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_neg.out ge converter is not implement!")


