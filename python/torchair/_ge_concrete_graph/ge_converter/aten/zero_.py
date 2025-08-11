from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.zero_.default)
def conveter_aten_zero__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::zero_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.zero_.default ge_converter is not implemented!")
