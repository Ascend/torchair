from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.quantized.elu.default)
def conveter_quantized_elu_default(
    self: Tensor,
    output_scale: float,
    output_zero_point: int,
    alpha: Union[Number, Tensor] = 1,
    scale: Union[Number, Tensor] = 1,
    input_scale: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: quantized::elu(Tensor self, float output_scale, int output_zero_point, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.elu.default ge_converter is not implemented!")
