from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest1d.vec)
def conveter_aten_upsample_nearest1d_vec(
    input: Tensor,
    output_size: Optional[Union[List[int], Tensor]],
    scale_factors: Optional[List[float]],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest1d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest1d.vec ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest1d.default)
def conveter_aten_upsample_nearest1d_default(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    scales: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::upsample_nearest1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest1d.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.upsample_nearest1d.out)
def conveter_aten_upsample_nearest1d_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    scales: Optional[float] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::upsample_nearest1d.out(Tensor self, SymInt[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.upsample_nearest1d.out ge_converter is not implemented!")
