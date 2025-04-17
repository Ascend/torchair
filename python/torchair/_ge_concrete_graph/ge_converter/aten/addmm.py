from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, _ge_dtype_to_ge_proto_dtype
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


def is_need_to_convert_bias(self, mat):
    """
    Use Matmul to speed up the calculation in the following conditions:
    Self dtype is float16 or bfloat16 and self rank is 1
    self dtype is float16 or bfloat16 and self rank is 2, but the first dimension of self is 1, 
    and the second dimension equal to the second dimension of mat2, for example: mat1: [5, 2], mat2:[2, 4], self: [1, 4]
    """
    if self.dtype == DataType.DT_FLOAT16 or self.dtype == DataType.DT_BF16:
        if self.rank == 1:
            return True
        elif self.rank == 2:
            return self.symsize[0] == 1 and self.symsize[1] == mat.symsize[1]
        return False
    return False


def get_addmm_output(self, beta, beta_is_zero, beta_is_one, mm_res):
    if beta_is_zero:
        output = mm_res
    elif beta_is_one:
        output = ge.Add(mm_res, self)
    else:
        output = ge.AxpyV2(mm_res, self, beta)
    return output


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
    alpha_is_one = True if not isinstance(alpha, Tensor) and float(alpha) == 1.0 else False
    beta_is_one = True if not isinstance(beta, Tensor) and float(beta) == 1.0 else False
    alpha_is_zero = True if not isinstance(alpha, Tensor) and float(alpha) == 0.0 else False
    beta_is_zero = True if not isinstance(beta, Tensor) and float(beta) == 0.0 else False

    mat1, mat2, self = dtype_promote(mat1, mat2, self, target_dtype=meta_outputs.dtype)
    alpha, beta = dtype_promote(alpha, beta, target_dtype=meta_outputs.dtype)
    
    # Case 1, alpha is 0
    if alpha_is_zero:
        return ge.Mul(self, beta)
    # Case 2, use matmul with bias
    if is_need_to_convert_bias(self, mat2):
        if not beta_is_one:
            self = ge.Mul(self, beta)
        if not alpha_is_one:
            mat1 = ge.Mul(mat1, alpha)

        return ge.MatMulV2(mat1, mat2, self, None)
    # Case 3, use mm+add/axpyv2
    else:
        mm_res = ge.MatMulV2(mat1, mat2, None, None)
        mm_res.desc.dtype = _ge_dtype_to_ge_proto_dtype(meta_outputs.dtype)
        # fp32 dtype need to reserve this Mul
        mm_res = ge.Mul(mm_res, alpha)
        return get_addmm_output(self, beta, beta_is_zero, beta_is_one, mm_res)


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
