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


@register_fx_node_ge_converter(torch.ops.prims.slice.default)
def conveter_prims_slice_default(
        a: Tensor,
        start_indices: Union[List[int], Tensor],
        limit_indices: Union[List[int], Tensor],
        strides: Optional[Union[List[int], Tensor]] = None,
        meta_outputs: Any = None):
    """ NB: prims::slice(Tensor(a) a, SymInt[] start_indices, SymInt[] limit_indices, SymInt[]? strides=None) -> Tensor(a) """
    raise NotImplementedError("torch.ops.prims.slice.default ge converter is not implement!")


