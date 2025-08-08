from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.quantized.dropout.default)
def conveter_quantized_dropout_default(
    self: Tensor,
    output_scale: float,
    output_zero_point: int,
    p: Union[Number, Tensor] = 0.5,
    training: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: quantized::dropout(Tensor self, float output_scale, int output_zero_point, Scalar p=0.5, bool training=False) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.dropout.default ge_converter is not implemented!")
