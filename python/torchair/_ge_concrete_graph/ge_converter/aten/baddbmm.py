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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(1, 3, 5), F32(1, 3, 4), F32(1, 4, 5)),
    Support(F32(1, 3, 5), F32(1, 3, 4), F32(1, 4, 5), beta=0.1, alpha=0.5),
    Support(F32(1, 3, 5), F32(1, 3, 4), F32(1, 4, 5), beta=0.0, alpha=1.0),
])
@register_fx_node_ge_converter(torch.ops.aten.baddbmm.default)
def conveter_aten_baddbmm_default(
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    batch_matmul_tensor = ge.BatchMatMul(batch1, batch2)
    alpha_mul_tensor = ge.Mul(batch_matmul_tensor, alpha)
    beta_mul_tensor = ge.Mul(self, beta)
    alpha_mul_tensor, beta_mul_tensor = dtype_promote(alpha_mul_tensor, beta_mul_tensor,
                                                      target_dtype=meta_outputs.dtype)
    return ge.Add(alpha_mul_tensor, beta_mul_tensor)


@register_fx_node_ge_converter(torch.ops.aten.baddbmm.out)
def conveter_aten_baddbmm_out(
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.baddbmm.out ge_converter is not implemented!")
