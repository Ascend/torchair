from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.t_.default)
def conveter_aten_t__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::t_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.t_.default ge_converter is not implemented!")
