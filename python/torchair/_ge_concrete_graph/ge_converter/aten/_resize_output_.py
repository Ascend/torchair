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
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._resize_output_.default)
def conveter_aten__resize_output__default(
    self: Tensor,
    size: Union[List[int], Tensor],
    device: Device,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_resize_output_(Tensor(a!) self, SymInt[] size, Device device) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._resize_output_.default ge_converter is not implemented!")
