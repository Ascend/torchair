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

npu = torch.ops.npu
aten = torch.ops.aten
prims = torch.ops.prims


@register_lowering(npu._npu_dtype_cast, type_promotion_kind=None)
@register_lowering(npu.npu_dtype_cast, type_promotion_kind=None)
def lowering_npu_dtype_cast(x: TensorBox, dtype: torch.dtype):
    return to_dtype(x, dtype, copy=True)
