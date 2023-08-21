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


@register_fx_node_ge_converter(torch.ops.prims.broadcast_in_dim.default)
def conveter_prims_broadcast_in_dim_default(
    a: Tensor,
    shape: Union[List[int], Tensor],
    broadcast_dimensions: List[int],
    meta_outputs: TensorSpec = None,
):
    """NB: prims::broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.broadcast_in_dim.default ge_converter is not implemented!")
