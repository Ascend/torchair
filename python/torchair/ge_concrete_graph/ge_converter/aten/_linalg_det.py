import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
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


@register_fx_node_ge_converter(torch.ops.aten._linalg_det.default)
def conveter_aten__linalg_det_default(
        A: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::_linalg_det(Tensor A) -> (Tensor result, Tensor LU, Tensor pivots) """
    raise NotImplementedError("torch.ops.aten._linalg_det.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._linalg_det.result)
def conveter_aten__linalg_det_result(
        A: Tensor,
        *,
        result: Tensor = None,
        LU: Tensor = None,
        pivots: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::_linalg_det.result(Tensor A, *, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots) """
    raise NotImplementedError("torch.ops.aten._linalg_det.result ge converter is not implement!")


