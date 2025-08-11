from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.is_pinned.default)
def conveter_aten_is_pinned_default(
    self: Tensor, device: Optional[Device] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::is_pinned(Tensor self, Device? device=None) -> bool"""
    raise NotImplementedError("torch.ops.aten.is_pinned.default ge_converter is not implemented!")
