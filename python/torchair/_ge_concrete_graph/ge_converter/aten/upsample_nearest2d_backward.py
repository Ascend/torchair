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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I64, Support
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, specific_op_output_layout


@declare_supported([
    Support(F32(1, 1, 4, 4), [4, 4], [1, 1, 2, 2], 2, 2),
    Support(F16(1, 1, 3, 3), [3, 3], [1, 1, 2, 2], 2, 2),
])
@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest2d_backward.default)
def conveter_aten_upsample_nearest2d_backward_default(
    grad_output: Tensor,
    output_size: Union[List[int], Tensor],
    input_size: Union[List[int], Tensor],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor"""
    if isinstance(input_size, Tensor):
        raise RuntimeError("torch.ops.aten.upsample_nearest2d_backward.default "
                           "is not implemented while input_size is Tensor!")

    grad_output = dtype_promote(grad_output, target_dtype=DataType.DT_FLOAT)
    output_sizes = ge.Const([input_size[2], input_size[3]], DataType.DT_INT32)
    out = ge.ResizeNearestNeighborV2Grad(grad_output, output_sizes, align_corners=False, half_pixel_centers=False)

    specific_op_input_layout(out, indices=0, layout="NCHW")
    specific_op_output_layout(out, indices=0, layout="NCHW")
    return ge.Cast(out, dst_type=meta_outputs.dtype)


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest2d_backward.grad_input)
def conveter_aten_upsample_nearest2d_backward_grad_input(
    grad_output: Tensor,
    output_size: Union[List[int], Tensor],
    input_size: Union[List[int], Tensor],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest2d_backward.grad_input ge_converter is not implemented!")
