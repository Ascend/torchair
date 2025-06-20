from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Optional
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu_inference.npu_moe_gating_top_k.default)
def conveter_npu_moe_gating_top_k_default(
        x: Tensor,
        k: int,
        *,
        bias: Optional[Tensor] = None,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        out_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        meta_outputs: List[TensorSpec] = None,
):
    return ge.MoeGatingTopK(x, bias, k=k, k_group=k_group, group_count=group_count, 
                            group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, 
                            out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)