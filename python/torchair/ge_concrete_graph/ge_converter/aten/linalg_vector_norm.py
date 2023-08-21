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


@register_fx_node_ge_converter(torch.ops.aten.linalg_vector_norm.default)
def conveter_aten_linalg_vector_norm_default(
    self: Tensor,
    ord: Union[Number, Tensor] = 2,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.linalg_vector_norm.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_vector_norm.out)
def conveter_aten_linalg_vector_norm_out(
    self: Tensor,
    ord: Union[Number, Tensor] = 2,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.linalg_vector_norm.out ge_converter is not implemented!")
