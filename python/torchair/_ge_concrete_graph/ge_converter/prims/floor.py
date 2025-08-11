from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.floor.default)
def conveter_prims_floor_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::floor(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.floor.default ge_converter is not implemented!")
