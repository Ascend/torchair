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


@register_fx_node_ge_converter(torch.ops.aten.huber_loss.default)
def conveter_aten_huber_loss_default(
        self: Tensor,
        target: Tensor,
        reduction: int = 1,
        delta: float = 1.0,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::huber_loss(Tensor self, Tensor target, int reduction=1, float delta=1.) -> Tensor """
    raise NotImplementedError("torch.ops.aten.huber_loss.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.huber_loss.out)
def conveter_aten_huber_loss_out(
        self: Tensor,
        target: Tensor,
        reduction: int = 1,
        delta: float = 1.0,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::huber_loss.out(Tensor self, Tensor target, int reduction=1, float delta=1., *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.huber_loss.out ge converter is not implement!")


