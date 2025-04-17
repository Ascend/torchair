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


@register_fx_node_ge_converter(torch.ops.aten._foreach_neg.default)
def conveter_aten__foreach_neg_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_neg(Tensor[] self) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_neg.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_neg.out)
def conveter_aten__foreach_neg_out(
    self: List[Tensor], *, out: List[Tensor] = None
):
    """NB: aten::_foreach_neg.out(Tensor[] self, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_neg.out ge_converter is not implemented!")
