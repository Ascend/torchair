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
from torchair._ge_concrete_graph.supported_declaration import F32, F16, BF16, Support


@declare_supported(
    [
        Support([F32(2, 2, 2), F16(2, 3), BF16(2, 3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_cosh.default)
def conveter_aten__foreach_cosh_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    return ge.ForeachCosh(self)
