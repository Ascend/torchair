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


@register_fx_node_ge_converter(torch.ops.aten.baddbmm.default)
def conveter_aten_baddbmm_default(
        self: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        *,
        beta: Union[Number, Tensor] = 1,
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.baddbmm.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.baddbmm.out)
def conveter_aten_baddbmm_out(
        self: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        *,
        beta: Union[Number, Tensor] = 1,
        alpha: Union[Number, Tensor] = 1,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.baddbmm.out ge converter is not implement!")


