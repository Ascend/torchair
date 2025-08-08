from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.quantized.leaky_relu.default)
def conveter_quantized_leaky_relu_default(
    qx: Tensor,
    negative_slope: Union[Number, Tensor],
    inplace: bool,
    output_scale: float,
    output_zero_point: int,
    meta_outputs: TensorSpec = None,
):
    """NB: quantized::leaky_relu(Tensor qx, Scalar negative_slope, bool inplace, float output_scale, int output_zero_point) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.leaky_relu.default ge_converter is not implemented!")
