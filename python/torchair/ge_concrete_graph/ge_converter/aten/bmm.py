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


@register_fx_node_ge_converter(torch.ops.aten.bmm.default)
def conveter_aten_bmm_default(
        self: Tensor,
        mat2: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::bmm(Tensor self, Tensor mat2) -> Tensor """
    return ge.BatchMatMul(self, mat2)


@register_fx_node_ge_converter(torch.ops.aten.bmm.out)
def conveter_aten_bmm_out(
        self: Tensor,
        mat2: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bmm.out ge converter is not implement!")


