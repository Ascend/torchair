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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F16(24, 1, 1, 128), F16(24, 1, 11, 128), F16(24, 1, 1, 128), F16(24, 1, 1, 128), layout='BSH'),
        Support(F32(24, 1, 1, 128), F32(24, 1, 11, 128), F32(24, 1, 1, 128), F32(24, 1, 1, 128), layout='BSH')
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_apply_rotary_pos_emb.default)
def conveter_npu_apply_rotary_pos_emb_default(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    layout: str = 'BSH',
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
    if layout != 'BSH':
        raise NotImplementedError("layout only support BSH now!")

    return ge.ApplyRotaryPosEmb(q, k, cos, sin, layout=1)