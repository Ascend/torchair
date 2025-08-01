from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import (
    Device,
    Number,
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
)
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import (
    register_fx_node_ge_converter,
    declare_supported,
)
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import (
    _TypedTensor,
    F32,
    BF16,
    F16,
    F64,
    I32,
    I16,
    I64,
    I8,
    U8,
    BOOL,
    Support,
)
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu.npu_rope_quant_kvcache.default)
def conveter_aten_rope_quant_kvcache_default(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    k_cache_ref: Tensor,
    v_cache_ref: Tensor,
    indices: Tensor,
    scale_k: Tensor,
    scale_v: Tensor,
    size_splits: List[int],
    offset_k: Optional[Tensor] = None,
    offset_v: Optional[Tensor] = None,
    quant_mode: int = 0,
    input_layout: str = "BSND",
    kv_output: bool = False,
    cache_mode: str = "contiguous",
    meta_outputs: List[TensorSpec] = None,
):
    quant_mode_str = "static" if quant_mode == 0 else "dynamic"
    return ge.DequantRopeQuantKvcache(
        x,
        cos,
        sin,
        k_cache_ref,
        v_cache_ref,
        indices,
        scale_k,
        scale_v,
        offset_k,
        offset_v,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        size_splits=size_splits,
        quant_mode=quant_mode_str,
        layout=input_layout,
        kv_output=kv_output,
        cache_mode=cache_mode,
    )
