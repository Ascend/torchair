import torch
import torch_npu

from torch._inductor.lowering import register_lowering, to_dtype
from torch._inductor.ir import (
    DtypeView,
    ExpandView,
    IndexingConstant,
    IRNode,
    is_triton,
    OnlineSoftmaxReduction,
    ops_wrapper,
    PermuteView,
    Pointwise,
    Reduction,
    SqueezeView,
    TensorBox,
    validate_ir,
    View,
)
from .common import _LoweringGuard, float_dtypes, byte_dtypes

npu = torch.ops.npu


@register_lowering(npu._npu_dtype_cast, type_promotion_kind=None)
@register_lowering(npu.npu_dtype_cast, type_promotion_kind=None)
def lowering_npu_dtype_cast(x: TensorBox, dtype: torch.dtype):
    return to_dtype(x, dtype, copy=True)


_LoweringGuard.support(torch.ops.npu._npu_dtype_cast, float_dtypes())
_LoweringGuard.support(torch.ops.npu.npu_dtype_cast, float_dtypes())
