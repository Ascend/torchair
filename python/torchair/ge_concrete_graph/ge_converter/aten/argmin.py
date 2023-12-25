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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote, DataType
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
     Support


@declare_supported([
     Support(F32(3, 2)),
     Support(F32(3, 2), 1),
     Support(F32(3, 2), 1, keepdim=True),
])
@register_fx_node_ge_converter(torch.ops.aten.argmin.default)
def conveter_aten_argmin_default(
    self: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    if not dim:
        self_cp = ge.Reshape(self, (-1,))
        return ge.ArgMin(self_cp, 0)
    if keepdim:
        index, _ = ge.ArgMinWithValue(self, dimension=dim, keep_dims=keepdim)
        index = dtype_promote(index, target_dtype=DataType.DT_FLOAT64)
        return index
    dim = dtype_promote(dim, target_dtype=DataType.DT_FLOAT64)
    return ge.ArgMin(self, dim)


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
