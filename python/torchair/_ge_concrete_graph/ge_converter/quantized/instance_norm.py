from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.quantized.instance_norm.default)
def conveter_quantized_instance_norm_default(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    output_scale: float,
    output_zero_point: int,
    meta_outputs: TensorSpec = None,
):
    """NB: quantized::instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.instance_norm.default ge_converter is not implemented!")
