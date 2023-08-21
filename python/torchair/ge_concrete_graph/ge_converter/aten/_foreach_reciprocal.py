from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._foreach_reciprocal.default)
def conveter_aten__foreach_reciprocal_default(
    self: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_reciprocal(Tensor[] self) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_reciprocal.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_reciprocal.out)
def conveter_aten__foreach_reciprocal_out(
    self: List[Tensor], *, out: List[Tensor] = None
):
    """NB: aten::_foreach_reciprocal.out(Tensor[] self, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_reciprocal.out ge_converter is not implemented!")
