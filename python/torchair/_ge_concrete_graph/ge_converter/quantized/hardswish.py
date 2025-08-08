from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.quantized.hardswish.default)
def conveter_quantized_hardswish_default(
    input: Tensor, output_scale: float, output_zero_point: int, meta_outputs: TensorSpec = None
):
    """NB: quantized::hardswish(Tensor input, float output_scale, int output_zero_point) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.hardswish.default ge_converter is not implemented!")
