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


@register_fx_node_ge_converter(torch.ops.aten.addmm.default)
def conveter_aten_addmm_default(
        self: Tensor,
        mat1: Tensor,
        mat2: Tensor,
        *,
        beta: Union[Number, Tensor] = 1,
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Any = None):
    """ NB: aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.addmm.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.addmm.out)
def conveter_aten_addmm_out(
        self: Tensor,
        mat1: Tensor,
        mat2: Tensor,
        *,
        beta: Union[Number, Tensor] = 1,
        alpha: Union[Number, Tensor] = 1,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.addmm.out ge converter is not implement!")


