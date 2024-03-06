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
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported(
    [
        Support(F16(4096), F16(2048, 4096), F16(4096, 4096)),
        Support(F32(2, 2), F32(2, 2), F32(2, 2)),
        Support(F32(2, 2), F32(2, 2), F32(2, 2), beta=2.1, alpha=3.2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.addmm.default)
def conveter_aten_addmm_default(
    self: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    # If self is float16 and alpha, beta are 1, use Matmul to speed up the calculation.
    if self.dtype == DataType.DT_FLOAT16 and float(alpha) == 1.0 and float(beta) == 1.0:
        mat1, mat2 = dtype_promote(mat1, mat2, target_dtype=meta_outputs.dtype)
        mm_res = ge.MatMul(mat1, mat2, self)
        return mm_res
    else:
        mat1, mat2 = dtype_promote(mat1, mat2, target_dtype=meta_outputs.dtype)
        mm_res = ge.MatMul(mat1, mat2, None)
        mm_res, self = dtype_promote(mm_res, self, target_dtype=meta_outputs.dtype)
        if not(not isinstance(alpha, Tensor) and alpha == 1):
            alpha = dtype_promote(alpha, target_dtype=meta_outputs.dtype)
            mm_res = ge.Mul(mm_res, alpha)
        if not(not isinstance(beta, Tensor) and beta == 1):
            beta = dtype_promote(beta, target_dtype=meta_outputs.dtype)
            self = ge.Mul(self, beta)
        output = ge.Add(mm_res, self)
        return output


@register_fx_node_ge_converter(torch.ops.aten.addmm.out)
def conveter_aten_addmm_out(
    self: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError(
        "torch.ops.aten.addmm.out is redundant before pytorch 2.1.0,might be supported in future version.")
