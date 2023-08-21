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


@register_fx_node_ge_converter(torch.ops.aten.as_strided.default)
def conveter_aten_as_strided_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    storage_offset: Optional[Union[int, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.as_strided.default ge_converter is not implemented!")
