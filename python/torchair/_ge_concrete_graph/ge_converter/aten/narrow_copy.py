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


@register_fx_node_ge_converter(torch.ops.aten.narrow_copy.default)
def conveter_aten_narrow_copy_default(
    self: Tensor,
    dim: int,
    start: Union[int, Tensor],
    length: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.narrow_copy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.narrow_copy.out)
def conveter_aten_narrow_copy_out(
    self: Tensor,
    dim: int,
    start: Union[int, Tensor],
    length: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::narrow_copy.out(Tensor self, int dim, SymInt start, SymInt length, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.narrow_copy.out ge_converter is not implemented!")
