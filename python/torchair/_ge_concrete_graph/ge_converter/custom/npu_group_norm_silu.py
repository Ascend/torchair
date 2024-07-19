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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(24, 320, 48, 48), F32(48), F32(48), group=32, eps=0.000100),
    Support(F16(1, 32, 128, 64), F16(32), F16(32), group=32, eps=0.000100),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_group_norm_silu.default)
def conveter_npu_group_norm_silu_default(
    self: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    group: int,
    eps: float = 0.000100,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_group_norm_silu(Tensor self, Tensor? weight, Tensor? bias, int group,
                                        float eps) -> (Tensor, Tensor, Tensor)"""
    return ge.GroupNormSilu(self, weight, bias, num_groups=group, eps=eps)