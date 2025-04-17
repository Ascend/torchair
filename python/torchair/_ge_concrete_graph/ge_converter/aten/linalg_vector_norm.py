from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


def calculate_p(p):
    if p == float(inf):
        return torch.iinfo(torch.int32).max
    elif p == float(-inf):
        return torch.iinfo(torch.int32).min
    else:
        return p


@declare_supported([
    Support(F32(4, 16), 2.0, [1], True),
    Support(F32(4, 16), 0.5, [1], True),
    Support(F32(4, 16), float(inf), [1], True),
    Support(F32(4, 16), float(-inf), [1], True),
    Support(F32(4, 16, 32), 2.0, [2], True),
    Support(F32(4, 16), 2.0, [1], False),
    Support(F32(4, 16), 2.0, None, False),
    Support(F16(4, 16), 2.0, [1], False),
    Support(F16(4, 16), 2.0, [1], False, dtype=torch.float32),
])
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
    pvalue = calculate_p(ord)
    result_axes = dim if dim else []
    result = ge.LpNormV2(self, p=pvalue, axes=result_axes, keepdim=keepdim, epsilon=0.0)
    result_dtype = dtype if dtype else self.dtype
    result_promote = dtype_promote(result, target_dtype=result_dtype)
    return result_promote


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
