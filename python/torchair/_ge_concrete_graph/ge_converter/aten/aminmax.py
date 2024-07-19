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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.aminmax.default)
def conveter_aten_aminmax_default(
    self: Tensor,
    *,
    dim: Optional[int] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)"""
    raise NotImplementedError("torch.ops.aten.aminmax.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.aminmax.out)
def conveter_aten_aminmax_out(
    self: Tensor,
    *,
    dim: Optional[int] = None,
    keepdim: bool = False,
    min: Tensor = None,
    max: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)"""
    raise NotImplementedError("torch.ops.aten.aminmax.out ge_converter is not implemented!")
