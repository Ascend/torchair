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


@register_fx_node_ge_converter(torch.ops.aten.t.default)
def conveter_aten_t_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::t(Tensor(a) self) -> Tensor(a) """
    if self.rank < 2:
        return ge.Identity(self)
    elif self.rank == 2:
        return ge.Transpose(self, [1, 0])
    else:
        raise NotImplementedError("torch.ops.aten.t.default unsupported case!")

