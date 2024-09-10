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

import sys
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge import attr
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, get_default_ge_graph
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported([
    Support(F16(4, 8192), F16(4, 8192), 131072),
    Support(F16(4, 8192), F16(4, 8192), 160000),
    Support(F32(4, 8192), F32(4, 8192), 1),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_prefetch.default)
def conveter_npu_scatter_update_default(
    self: Tensor,
    dependency: Optional[Tensor],
    max_size: int,
    meta_outputs: TensorSpec = None,
):
    """NB: func: npu_prefetch(Tensor self, Tensor? dependency, int max_size) -> ()"""
    if max_size <= 0:
        raise ValueError(f"max_size should be greater than zero, but got {max_size}")
    if dependency is None:
        raise NotImplementedError("torch.ops.npu.npu_prefetch.default ge converter is not implement "
                                  "when dependency is None.")
    ge.Cmo(self, max_size=max_size, dependencies=[dependency])
    