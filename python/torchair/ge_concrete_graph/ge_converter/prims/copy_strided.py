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


@register_fx_node_ge_converter(torch.ops.prims.copy_strided.default)
def conveter_prims_copy_strided_default(
        a: Tensor,
        stride: Union[List[int], Tensor],
        meta_outputs: Any = None):
    """ NB: prims::copy_strided(Tensor a, SymInt[] stride) -> Tensor """
    raise NotImplementedError("torch.ops.prims.copy_strided.default ge converter is not implement!")


