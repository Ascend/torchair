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


@register_fx_node_ge_converter(torch.ops.prims.iota.default)
def conveter_prims_iota_default(
    length: Union[int, Tensor],
    *,
    start: Union[int, Tensor],
    step: Union[int, Tensor],
    dtype: int,
    device: Device,
    requires_grad: bool,
    meta_outputs: TensorSpec = None
):
    """NB: prims::iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.iota.default ge_converter is not implemented!")
