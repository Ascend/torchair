from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.heaviside_.default)
def conveter_aten_heaviside__default(
    self: Tensor, values: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.heaviside_.default ge_converter is not implemented!")
