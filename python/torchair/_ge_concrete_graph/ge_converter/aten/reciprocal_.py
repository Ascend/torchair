from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.reciprocal_.default)
def conveter_aten_reciprocal__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.reciprocal_.default ge_converter is not implemented!")
