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


@register_fx_node_ge_converter(torch.ops.prims.uniform.default)
def conveter_prims_uniform_default(
    shape: Union[List[int], Tensor],
    *,
    low: Union[Number, Tensor],
    high: Union[Number, Tensor],
    dtype: int,
    device: Device,
    meta_outputs: TensorSpec = None
):
    """NB: prims::uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.uniform.default ge_converter is not implemented!")
