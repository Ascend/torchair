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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.argmin.default)
def conveter_aten_argmin_default(
    self: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.argmin.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.argmin.out)
def conveter_aten_argmin_out(
    self: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.argmin.out ge_converter is not implemented!")
