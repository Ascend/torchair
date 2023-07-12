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


@register_fx_node_ge_converter(torch.ops.aten.softplus.default)
def conveter_aten_softplus_default(
        self: Tensor,
        beta: Union[Number, Tensor] = 1,
        threshold: Union[Number, Tensor] = 20,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor """
    raise NotImplementedError("torch.ops.aten.softplus.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.softplus.out)
def conveter_aten_softplus_out(
        self: Tensor,
        beta: Union[Number, Tensor] = 1,
        threshold: Union[Number, Tensor] = 20,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.softplus.out ge converter is not implement!")


