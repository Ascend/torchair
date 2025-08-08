from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.quantized.layer_norm.default)
def conveter_quantized_layer_norm_default(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    output_scale: float,
    output_zero_point: int,
    meta_outputs: TensorSpec = None,
):
    """NB: quantized::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.layer_norm.default ge_converter is not implemented!")
