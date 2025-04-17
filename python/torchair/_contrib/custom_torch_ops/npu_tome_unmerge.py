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
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support, I64


@declare_supported(
    [
        Support(F16(32, 32, 32), I64(32), I64(32), I64(32), I64(32), float=0.5),
    ]
)

@register_fx_node_ge_converter(torch.ops.npu_inference.npu_tome_unmerge.default)
def conveter_npu_tome_unmerge_default(
    atten_out: Tensor,
    ori_indice_a: Tensor,
    ori_indice_b: Tensor,
    topk_indice: Tensor,
    arg_max: Tensor,
    top_r_rate: float = 0.5,
    meta_outputs: TensorSpec = None,
):
    """NB: func: npu_tome_unmerge(Tensor atten_out, Tensor ori_indice_a, 
        Tensor ori_indice_b, Tensor topk_indice, Tensor arg_max, float top_r_rate) -> Tensor
    """
    return ge.TomeUnmerge(atten_out, ori_indice_a, ori_indice_b, topk_indice, arg_max, top_r_rate=top_r_rate)