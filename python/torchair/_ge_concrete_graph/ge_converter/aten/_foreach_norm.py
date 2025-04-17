from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, F16, BF16, I32, Support


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], 1.),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], 1.),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], 1.),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_norm.Scalar)
def conveter_aten__foreach_norm_scalar(
    self: List[Tensor],
    scalar: Union[Number, Tensor] = 2,
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_norm.Scalar(Tensor[] self, Union[Number, Tensor] scalar) -> Tensor[]"""
    return ge.ForeachNorm(self, scalar)

