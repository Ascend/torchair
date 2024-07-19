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


@register_fx_node_ge_converter(torch.ops.aten.softplus.default)
def conveter_aten_softplus_default(
    self: Tensor,
    beta: Union[Number, Tensor] = 1,
    threshold: Union[Number, Tensor] = 20,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.softplus.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.softplus.out)
def conveter_aten_softplus_out(
    self: Tensor,
    beta: Union[Number, Tensor] = 1,
    threshold: Union[Number, Tensor] = 20,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.softplus.out ge_converter is not implemented!")
