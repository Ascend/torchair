from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._foreach_sqrt.default)
def conveter_aten__foreach_sqrt_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_sqrt(Tensor[] self) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_sqrt.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_sqrt.out)
def conveter_aten__foreach_sqrt_out(
    self: List[Tensor], *, out: List[Tensor] = None
):
    """NB: aten::_foreach_sqrt.out(Tensor[] self, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_sqrt.out ge_converter is not implemented!")
