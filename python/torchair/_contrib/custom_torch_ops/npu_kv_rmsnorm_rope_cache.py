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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8,\
    U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu_inference.npu_kv_rmsnorm_rope_cache.default)
def conveter_npu_kv_rmsnorm_rope_cache_default(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    epsilon: float = 1e-5,
    meta_outputs: List[TensorSpec] = None

):
    return ge.KvRmsNormRopeCache(kv, gamma, cos, sin, index, k_cache, v_cache, epsilon=epsilon)
