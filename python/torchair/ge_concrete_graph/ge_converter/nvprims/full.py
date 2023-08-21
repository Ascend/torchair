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


@register_fx_node_ge_converter(torch.ops.nvprims.full.default)
def conveter_nvprims_full_default(
    size: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    requires_grad: Optional[bool] = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: nvprims::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool? requires_grad=None) -> Tensor"""
    raise NotImplementedError("torch.ops.nvprims.full.default ge_converter is not implemented!")
