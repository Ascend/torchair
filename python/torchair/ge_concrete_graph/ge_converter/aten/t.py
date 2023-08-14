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


@register_fx_node_ge_converter(torch.ops.aten.t.default)
def conveter_aten_t_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::t(Tensor(a) self) -> Tensor(a)"""
    if self.rank < 2:
        return ge.Identity(self)
    elif self.rank == 2:
        return ge.Transpose(self, [1, 0])
    else:
        raise NotImplementedError("torch.ops.aten.t.default unsupported case!")
