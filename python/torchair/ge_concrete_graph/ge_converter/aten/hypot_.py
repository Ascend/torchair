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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.hypot_.default)
def conveter_aten_hypot__default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.hypot_.default ge_converter is not implemented!")
