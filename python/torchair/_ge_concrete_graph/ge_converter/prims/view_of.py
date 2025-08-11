from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.view_of.default)
def conveter_prims_view_of_default(a: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::view_of(Tensor(a) a) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.view_of.default ge_converter is not implemented!")
