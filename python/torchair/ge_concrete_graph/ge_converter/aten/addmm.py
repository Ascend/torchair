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


def is_support_nd_out():
    return True if torch.npu.utils.get_soc_version() >= 220 else False


@declare_supported(
    [
        Support(F16(4096), F16(2048, 4096), F16(4096, 4096)),
        Support(F32(768), F32(2048, 3072), F32(3072, 768)),
        Support(F16(768), F16(2048, 3072), F16(3072, 768)),
        Support(F32(2, 3), F32(2, 3), F32(3, 3)),
        Support(F32(2, 2), F32(2, 2), F32(2, 2), beta=2.1, alpha=3.2),
        Support(F32(2, 2), F32(2, 2), F32(2, 2), beta=2.1, alpha=1),
        Support(F32(2, 2), F32(2, 2), F32(2, 2), beta=1, alpha=3.2),
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
    # The dtype of alpha and beta will be prompted if necessary, so check their value at first
    alpha_is_one = True if float(alpha) == 1.0 else False
    beta_is_one = True if float(beta) == 1.0 else False

    mat1, mat2, self = dtype_promote(mat1, mat2, self, target_dtype=meta_outputs.dtype)
    alpha, beta = dtype_promote(alpha, beta, target_dtype=meta_outputs.dtype)
    
    # If self dtype is float16 and self rank is 1, use Matmul to speed up the calculation.
    self_rank = self.rank
    if_support_nd_out = is_support_nd_out()
    # Case 1, use matmul with bias
    if self.dtype == DataType.DT_FLOAT16 and self_rank == 1 and if_support_nd_out:
        if not beta_is_one:
            self = ge.Mul(self, beta)
        if not alpha_is_one:
            mat1 = ge.Mul(mat1, alpha)

        return ge.MatMul(mat1, mat2, self)
    # Case 2, use mm+add/axpyv2
    else:
        mm_res = ge.MatMul(mat1, mat2, None)
        mm_res = dtype_promote(mm_res, target_dtype=meta_outputs.dtype)
        if not(not isinstance(alpha, Tensor) and alpha == 1):
            mm_res = ge.Mul(mm_res, alpha)
        if beta_is_one:
            output = ge.Add(mm_res, self)
        else:
            output = ge.AxpyV2(mm_res, self, beta)
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
