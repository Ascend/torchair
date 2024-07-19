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


@register_fx_node_ge_converter(torch.ops.npu.scatter_update.default)
def conveter_npu_scatter_update_default(
    data: Tensor,
    indices: Tensor,
    updates: Tensor = 0,
    axis: int = 0,
    meta_outputs: TensorSpec = None,
):
    """NB: func: scatter_update(Tensor data, Tensor indices, Tensor updates, int axis) -> Tensor"""
    """
    Warning: kernel [scatter_update] is a out-of-place op, but it is supported by another in-place op cann.Scatter.
    This current usage may cause the input to be changed unexpectedly, 
    and the caller needs to pay attention to this feature.
    """

    copy = ge.TensorMove(data)
    return ge.Scatter(copy, indices, updates, reduce="update", axis=axis)
