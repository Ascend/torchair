from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, I32, Support


@declare_supported([
    Support(I32(8, ), num_experts=8)
])
@register_fx_node_ge_converter(torch.ops.npu.npu_moe_compute_expert_tokens.default)
def conveter_moe_compute_expert_tokens(
    sorted_experts: Tensor,
    num_experts: int,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_moe_compute_expert_tokens(Tensor sorted_experts, int32 num_experts) -> Tensor"""
    return ge.MoeComputeExpertTokens(sorted_experts, num_experts=num_experts)
