from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.is_coalesced.default)
def conveter_aten_is_coalesced_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::is_coalesced(Tensor self) -> bool"""
    raise NotImplementedError("torch.ops.aten.is_coalesced.default ge_converter is not implemented!")
