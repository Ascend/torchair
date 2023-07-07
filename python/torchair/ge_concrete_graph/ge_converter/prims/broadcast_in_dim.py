import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


@register_fx_node_ge_converter(torch.ops.prims.broadcast_in_dim.default)
def conveter_prims_broadcast_in_dim_default(
        a: Tensor,
        shape: Union[List[int], Tensor],
        broadcast_dimensions: List[int],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: prims::broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a) """
    raise NotImplementedError("torch.ops.prims.broadcast_in_dim.default ge converter is not implement!")


