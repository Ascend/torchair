from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.relu6.default)
def conveter_aten_relu6_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::relu6(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.relu6.default ge_converter is not implemented!")
