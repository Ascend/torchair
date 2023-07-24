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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._log_softmax.default)
def conveter_aten__log_softmax_default(
    self: Tensor, dim: int, half_to_float: bool, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._log_softmax.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._log_softmax.out)
def conveter_aten__log_softmax_out(
    self: Tensor,
    dim: int,
    half_to_float: bool,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._log_softmax.out ge_converter is not implemented!")
