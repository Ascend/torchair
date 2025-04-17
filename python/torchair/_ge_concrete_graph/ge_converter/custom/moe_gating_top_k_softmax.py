from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import sys
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, Support, I32, BOOL


@declare_supported(
    [
        Support(F32(3, 4), BOOL(3), int=1),
    ]
)

@register_fx_node_ge_converter(torch.ops.npu.npu_moe_gating_top_k_softmax.default)
def conveter_npu_moe_gating_top_k_softmax_default(
        x: Tensor,
        finished: Optional[Tensor] = None,
        k: int = 1,
        meta_outputs: TensorSpec = None,
):
    """NB: func: npu_moe_init_routing(Tensor x, Tensor row_idx, Tensor expert_idx, int active_num)
    -> (Tensor, Tensor, Tensor)
    """
    return ge.MoeGatingTopKSoftmax(x, finished, k=k)
