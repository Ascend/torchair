from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.item.default)
def conveter_aten_item_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::item(Tensor self) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.item.default ge_converter is not implemented!")
