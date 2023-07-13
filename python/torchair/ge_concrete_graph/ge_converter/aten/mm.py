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


@register_fx_node_ge_converter(torch.ops.aten.mm.default)
def conveter_aten_mm_default(
        self: Tensor,
        mat2: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::mm(Tensor self, Tensor mat2) -> Tensor """
    return ge.MatMul(self, mat2, None)


@register_fx_node_ge_converter(torch.ops.aten.mm.out)
def conveter_aten_mm_out(
        self: Tensor,
        mat2: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.mm.out ge converter is not implement!")


