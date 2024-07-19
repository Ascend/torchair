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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.logit.default)
def conveter_aten_logit_default(
    self: Tensor, eps: Optional[float] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logit(Tensor self, float? eps=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logit.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logit.out)
def conveter_aten_logit_out(
    self: Tensor,
    eps: Optional[float] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logit.out ge_converter is not implemented!")
