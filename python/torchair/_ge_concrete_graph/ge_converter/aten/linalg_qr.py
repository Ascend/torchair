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


@register_fx_node_ge_converter(torch.ops.aten.linalg_qr.default)
def conveter_aten_linalg_qr_default(
    A: Tensor, mode: str = "reduced", meta_outputs: List[TensorSpec] = None
):
    """NB: aten::linalg_qr(Tensor A, str mode="reduced") -> (Tensor Q, Tensor R)"""
    raise NotImplementedError("torch.ops.aten.linalg_qr.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_qr.out)
def conveter_aten_linalg_qr_out(
    A: Tensor,
    mode: str = "reduced",
    *,
    Q: Tensor = None,
    R: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::linalg_qr.out(Tensor A, str mode="reduced", *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)"""
    raise NotImplementedError("torch.ops.aten.linalg_qr.out ge_converter is not implemented!")
