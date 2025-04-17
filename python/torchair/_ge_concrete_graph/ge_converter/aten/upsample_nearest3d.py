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


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest3d.vec)
def conveter_aten_upsample_nearest3d_vec(
    input: Tensor,
    output_size: Optional[Union[List[int], Tensor]],
    scale_factors: Optional[List[float]],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest3d.vec ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest3d.default)
def conveter_aten_upsample_nearest3d_default(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest3d.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest3d.out)
def conveter_aten_upsample_nearest3d_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::upsample_nearest3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest3d.out ge_converter is not implemented!")
