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


@register_fx_node_ge_converter(torch.ops.aten.diagonal.default)
def conveter_aten_diagonal_default(
    self: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.diagonal.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal.Dimname)
def conveter_aten_diagonal_Dimname(
    self: Tensor,
    *,
    outdim: str,
    dim1: str,
    dim2: str,
    offset: int = 0,
    meta_outputs: TensorSpec = None
):
    """NB: aten::diagonal.Dimname(Tensor(a) self, *, str outdim, str dim1, str dim2, int offset=0) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.diagonal.Dimname ge_converter is not implemented!")
