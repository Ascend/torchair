from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.leaky_relu_.default)
def conveter_aten_leaky_relu__default(
    self: Tensor, negative_slope: Union[Number, Tensor] = 0.01, meta_outputs: TensorSpec = None
):
    """NB: aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.leaky_relu_.default ge_converter is not implemented!")
