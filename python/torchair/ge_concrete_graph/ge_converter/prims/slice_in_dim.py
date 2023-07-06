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


@register_fx_node_ge_converter(torch.ops.prims.slice_in_dim.default)
def conveter_prims_slice_in_dim_default(
        a: Tensor,
        start_index: Union[int, Tensor],
        limit_index: Union[int, Tensor],
        stride: int = 1,
        axis: int = 0,
        meta_outputs: Any = None):
    """ NB: prims::slice_in_dim(Tensor(a) a, SymInt start_index, SymInt limit_index, int stride=1, int axis=0) -> Tensor(a) """
    raise NotImplementedError("torch.ops.prims.slice_in_dim.default ge converter is not implement!")


