from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.log_.default)
def conveter_aten_log__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log_.default ge_converter is not implemented!")
