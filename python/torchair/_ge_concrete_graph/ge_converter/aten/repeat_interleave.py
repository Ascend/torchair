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


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.Tensor)
def conveter_aten_repeat_interleave_Tensor(
    repeats: Tensor, *, output_size: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.self_Tensor)
def conveter_aten_repeat_interleave_self_Tensor(
    self: Tensor,
    repeats: Tensor,
    dim: Optional[int] = None,
    *,
    output_size: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.self_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.self_int)
def conveter_aten_repeat_interleave_self_int(
    self: Tensor,
    repeats: Union[int, Tensor],
    dim: Optional[int] = None,
    *,
    output_size: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, int? output_size=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.self_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.Tensor_out)
def conveter_aten_repeat_interleave_Tensor_out(
    repeats: Tensor,
    *,
    output_size: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.Tensor_out(Tensor repeats, *, int? output_size=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.Tensor_out ge_converter is not implemented!")
