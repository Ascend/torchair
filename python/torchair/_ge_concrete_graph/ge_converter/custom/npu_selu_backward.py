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


@register_fx_node_ge_converter(torch.ops.npu.selu_backward.default)
def conveter_npu_selu_backward_default(
    self: Tensor,
    result: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_selu_backward_default(Tensor self, Tensor result) -> Tensor"""
    return ge.SeluGrad(self, result)
