import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten._resize_output_.default)
def conveter_aten__resize_output__default(
        self: Tensor,
        size: Union[List[int], Tensor],
        device: Device,
        meta_outputs: Any = None):
    """ NB: aten::_resize_output_(Tensor(a!) self, SymInt[] size, Device device) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._resize_output_.default ge converter is not implement!")


