from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.item.default)
def conveter_prims_item_default(a: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::item(Tensor a) -> Scalar"""
    raise NotImplementedError("torch.ops.prims.item.default ge_converter is not implemented!")
