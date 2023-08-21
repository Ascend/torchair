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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.broadcast_tensors.default)
def conveter_aten_broadcast_tensors_default(
    tensors: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.broadcast_tensors.default ge_converter is not implemented!")
