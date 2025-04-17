from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.is_same_size.default)
def conveter_aten_is_same_size_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::is_same_size(Tensor self, Tensor other) -> bool"""
    raise NotImplementedError("torch.ops.aten.is_same_size.default ge_converter is not implemented!")
