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


@register_fx_node_ge_converter(torch.ops.aten.fill_.Scalar)
def conveter_aten_fill__Scalar(
    self: Tensor, value: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fill_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fill_.Tensor)
def conveter_aten_fill__Tensor(self: Tensor, value: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fill_.Tensor ge_converter is not implemented!")
