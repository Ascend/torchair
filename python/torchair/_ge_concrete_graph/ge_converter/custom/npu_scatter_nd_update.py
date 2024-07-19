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
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported([
    Support(F32(3, 3, 3), F32(2, 3), F32(2)),
    Support(F32(3, 3, 3), F32(2, 3), F32(2)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update.default)
def conveter_npu_scatter_nd_update_default(
    self: Tensor,
    indices: Tensor,
    updates: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: func: scatter_nd_update(Tensor self, Tensor indices, Tensor updates) -> Tensor"""

    copy = ge.TensorMove(self)
    return ge.ScatterNdUpdate(copy, indices, updates)
