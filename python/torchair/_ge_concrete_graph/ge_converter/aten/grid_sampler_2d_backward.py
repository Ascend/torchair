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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(6, 3, 3, 3), F32(6, 3, 3, 3), F32(6, 3, 3, 2), 0, 0, False, [False, True]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d_backward.default)
def conveter_aten_grid_sampler_2d_backward_default(
    grad_output: Tensor,
    input: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: List[bool],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)"""
    if not (0 <= interpolation_mode <= 2):
        raise AssertionError("interpolation_mode must be in range [0~2]")
    if not (0 <= padding_mode <= 2):
        raise AssertionError("padding_mode must be in range [0~2]")
    grad_output_cp = grad_output
    input_cp = input
    grid_cp = grid
    if grad_output.desc.dtype == ProtoDataType.DT_FLOAT16:
        grad_output_cp = dtype_promote(grad_output, target_dtype=DataType.DT_FLOAT)
    if input.desc.dtype == ProtoDataType.DT_FLOAT16:
        input_cp = dtype_promote(input, target_dtype=DataType.DT_FLOAT)
    if grid.desc.dtype == ProtoDataType.DT_FLOAT16:
        grid_cp = dtype_promote(grid, target_dtype=DataType.DT_FLOAT)
    inter_mode = ["bilinear", "nearest", "bicubic"]
    pad_mode = ["zeros", "border", "reflection"]
    dx, dgrid = ge.GridSampler2DGrad(grad_output_cp, input_cp, grid_cp, 
                                    interpolation_mode=inter_mode[interpolation_mode],
                                    padding_mode=pad_mode[padding_mode], align_corners=align_corners)
    if dx.dtype != input.dtype:
        dx = dtype_promote(dx, target_dtype=input.dtype)
        dgrid = dtype_promote(dgrid, target_dtype=input.dtype)
    return dx, dgrid


@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d_backward.out)
def conveter_aten_grid_sampler_2d_backward_out(
    grad_output: Tensor,
    input: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: List[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::grid_sampler_2d_backward.out(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"""
    raise RuntimeError("torch.ops.aten.grid_sampler_2d_backward.out ge_converter is not supported!")
