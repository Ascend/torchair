from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, \
    U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote

@register_fx_node_ge_converter(torch.ops.npu.npu_kv_rmsnorm_rope_cache.default)
def conveter_npu_kv_rmsnorm_rope_cache_default(
    kv: Tensor,
    gamma: Tensor,
    cos: Tensor,
    sin: Tensor,
    index: Tensor,
    k_cache: Tensor,
    ckv_cache: Tensor,
    *,
    k_rope_scale: Optional[Tensor] = None,
    c_kv_scale: Optional[Tensor] = None,
    k_rope_offset: Optional[Tensor] = None,
    c_kv_offset: Optional[Tensor] = None,
    epsilon: float = 1e-5,
    cache_mode: str = 'Norm',
    is_output_kv: bool = False,
    meta_outputs: List[TensorSpec] = None

):
    return ge.KvRmsNormRopeCache(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                 k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                 k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset,
                                 epsilon=epsilon, cache_mode=cache_mode, is_output_kv=is_output_kv)
