from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 2, 3, 3, 3), F32(2, 3, 3, 3, 3), 0, 0, False),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_3d.default)
def conveter_aten_grid_sampler_3d_default(
    input_tensor: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    meta_outputs: TensorSpec = None,
):
    if not (0 <= interpolation_mode <= 1):
        raise AssertionError("interpolation_mode must be in range [0~1]")
    if not (0 <= padding_mode <= 2):
        raise AssertionError("padding_mode must be in range [0~2]")
    input_cp = input_tensor
    grid_cp = grid
    if input_tensor.desc.dtype == ProtoDataType.DT_FLOAT16:
        input_cp = dtype_promote(input_tensor, target_dtype=DataType.DT_FLOAT)
    if grid.desc.dtype == ProtoDataType.DT_FLOAT16:
        grid_cp = dtype_promote(grid, target_dtype=DataType.DT_FLOAT)
    inter_modes = ["bilinear", "nearest"]
    pad_modes = ["zeros", "border", "reflection"]
    result = ge.GridSampler3D(input_cp, grid_cp, interpolation_mode=inter_modes[interpolation_mode],
                             padding_mode=pad_modes[padding_mode], align_corners=align_corners)
    if result.desc.dtype != input_tensor.desc.dtype:
        result = dtype_promote(result, target_dtype=input_tensor.dtype)
    return result


@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_3d.out)
def conveter_aten_grid_sampler_3d_out(
    input_tensor: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    raise RuntimeError("torch.ops.aten.grid_sampler_3d.out ge_converter is not supported!")
