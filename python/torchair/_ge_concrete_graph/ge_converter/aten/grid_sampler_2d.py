from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 2, 3, 3), F32(2, 3, 3, 2), 0, 0, False),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d.default)
def conveter_aten_grid_sampler_2d_default(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""
    if not (0 <= interpolation_mode <= 2):
        raise AssertionError("interpolation_mode must be in range [0~2]")
    if not (0 <= padding_mode <= 2):
        raise AssertionError("padding_mode must be in range [0~2]")
    input_cp = input
    grid_cp = grid
    if input.desc.dtype == ProtoDataType.DT_FLOAT16:
        input_cp = dtype_promote(input, target_dtype=DataType.DT_FLOAT)
    if grid.desc.dtype == ProtoDataType.DT_FLOAT16:
        grid_cp = dtype_promote(grid, target_dtype=DataType.DT_FLOAT)
    inter_modes = ["bilinear", "nearest", "bicubic"]
    pad_modes = ["zeros", "border", "reflection"]
    result = ge.GridSampler2D(input_cp, grid_cp, interpolation_mode=inter_modes[interpolation_mode],
                             padding_mode=pad_modes[padding_mode], align_corners=align_corners)
    if result.desc.dtype != input.desc.dtype:
        result = dtype_promote(result, target_dtype=input.dtype)
    return result


@register_fx_node_ge_converter(torch.ops.aten.grid_sampler_2d.out)
def conveter_aten_grid_sampler_2d_out(
    input: Tensor,
    grid: Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::grid_sampler_2d.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.grid_sampler_2d.out ge_converter is not supported!")
