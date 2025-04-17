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


@register_fx_node_ge_converter(torch.ops.aten._upsample_bilinear2d_aa.default)
def conveter_aten__upsample_bilinear2d_aa_default(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_upsample_bilinear2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._upsample_bilinear2d_aa.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._upsample_bilinear2d_aa.out)
def conveter_aten__upsample_bilinear2d_aa_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_upsample_bilinear2d_aa.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._upsample_bilinear2d_aa.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._upsample_bilinear2d_aa.vec)
def conveter_aten__upsample_bilinear2d_aa_vec(
    input: Tensor,
    output_size: Optional[Union[List[int], Tensor]],
    align_corners: bool,
    scale_factors: Optional[List[float]],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_upsample_bilinear2d_aa.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._upsample_bilinear2d_aa.vec ge_converter is not implemented!")
