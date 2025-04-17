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


@register_fx_node_ge_converter(torch.ops.prims.slice.default)
def conveter_prims_slice_default(
    a: Tensor,
    start_indices: Union[List[int], Tensor],
    limit_indices: Union[List[int], Tensor],
    strides: Optional[Union[List[int], Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: prims::slice(Tensor(a) a, SymInt[] start_indices, SymInt[] limit_indices, SymInt[]? strides=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.slice.default ge_converter is not implemented!")
