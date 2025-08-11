from torchair._ge_concrete_graph.ge_converter.converter_utils import *


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
