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


@register_fx_node_ge_converter(torch.ops.aten._linalg_det.default)
def conveter_aten__linalg_det_default(A: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::_linalg_det(Tensor A) -> (Tensor result, Tensor LU, Tensor pivots)"""
    raise NotImplementedError("torch.ops.aten._linalg_det.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._linalg_det.result)
def conveter_aten__linalg_det_result(
    A: Tensor,
    *,
    result: Tensor = None,
    LU: Tensor = None,
    pivots: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_linalg_det.result(Tensor A, *, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots)"""
    raise NotImplementedError("torch.ops.aten._linalg_det.result ge_converter is not implemented!")
