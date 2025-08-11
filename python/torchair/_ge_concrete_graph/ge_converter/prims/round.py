from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.round.default)
def conveter_prims_round_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::round(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.round.default ge_converter is not implemented!")
