from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, specific_op_output_layout


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest2d.vec)
def conveter_aten_upsample_nearest2d_vec(
    input: Tensor,
    output_size: Optional[Union[List[int], Tensor]],
    scale_factors: Optional[List[float]],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest2d.vec ge_converter is not implemented!")


@declare_supported([
    Support(F16(2, 1280, 8, 8), [16, 16], 2.0, 2.0),
    Support(F16(2, 1280, 16, 16), [32, 32], 2.0, 2.0),
    Support(F16(2, 1280, 8, 8), [32, 32], 4.0, 4.0),
])
@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest2d.default)
def conveter_aten_upsample_nearest2d_default(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor"""
    output_size = dtype_promote(output_size, target_dtype=DataType.DT_INT32)
    out = ge.ResizeNearestNeighborV2(self, output_size, align_corners=False, half_pixel_centers=False)
    specific_op_input_layout(out, indices=0, layout="NCHW")
    specific_op_output_layout(out, indices=0, layout="NCHW")
    return out


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest2d.out)
def conveter_aten_upsample_nearest2d_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::upsample_nearest2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest2d.out ge_converter is not implemented!")
