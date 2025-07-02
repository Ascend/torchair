from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import os
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.utils import get_cann_opp_version


@register_fx_node_ge_converter(torch.ops.npu.npu_apply_rotary_pos_emb.default)
def conveter_npu_apply_rotary_pos_emb_default(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    layout: str = 'BSH',
    rotary_mode: str = 'half',
    meta_outputs: List[TensorSpec] = None,
):
    """
    NB: npu::npu_apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, str layout='BSH') 
                                      -> (Tensor, Tensor)
    """
    """
    Warning: kernel [npu_apply_rotary_pos_emb] is not a out-of-place op, but used as in-place op.
             This current usage may cause the input to be changed unexpectedly, and the caller 
             needs to pay attention to this feature.
    """
    support_layer = ['BSH', 'BSND', 'SBND', 'BNSD']
    support_rotary_mode = ['half', 'quarter', 'interleave']
    if layout not in support_layer:
        raise NotImplementedError("layout only support BSH/BSND/SBND/BNSD now!")
    if rotary_mode not in support_rotary_mode:
        raise NotImplementedError("rotary_mode only support half/quarter/interleave now!")

    layout_val = 1
    if layout == "SBND":
        layout_val = 2
    elif layout == "BNSD":
        layout_val = 3

    version_list = ["7.2", "7.3"]
    opp_ver = get_cann_opp_version()
    insert_tensor_move = True
    for ver in version_list:
        if opp_ver.startswith(ver):
            insert_tensor_move = False
            break
    if insert_tensor_move:
        tm_q = ge.TensorMove(query)
        tm_k = ge.TensorMove(key)
        return ge.ApplyRotaryPosEmb(tm_q, tm_k, cos, sin, layout=layout_val, rotary_mode=rotary_mode)
    else:
        return ge.ApplyRotaryPosEmb(query, key, cos, sin, layout=layout_val, rotary_mode=rotary_mode)
