import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


@register_fx_node_ge_converter(torch.ops.aten.addmv.default)
def conveter_aten_addmv_default(
        self: Tensor,
        mat: Tensor,
        vec: Tensor,
        *,
        beta: Union[Number, Tensor] = 1,
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.addmv.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.addmv.out)
def conveter_aten_addmv_out(
        self: Tensor,
        mat: Tensor,
        vec: Tensor,
        *,
        beta: Union[Number, Tensor] = 1,
        alpha: Union[Number, Tensor] = 1,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.addmv.out ge converter is not implement!")


