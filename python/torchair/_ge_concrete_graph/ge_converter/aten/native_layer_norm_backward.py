from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.native_layer_norm_backward.default)
def conveter_aten_native_layer_norm_backward_default(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: Union[List[int], Tensor],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: List[bool],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""
    if weight is None:
        weight = ge.Fill(ge.Const(normalized_shape, dtype=DataType.DT_INT32), 
                         ge.Cast(1., dst_type=input.dtype))
    return ge.LayerNormGradV3(grad_out, input, rstd, mean, weight)


@register_fx_node_ge_converter(torch.ops.aten.native_layer_norm_backward.out)
def conveter_aten_native_layer_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: Union[List[int], Tensor],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: List[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_layer_norm_backward.out(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise RuntimeError("torch.ops.aten.native_layer_norm_backward.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
