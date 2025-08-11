from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.log.default)
def conveter_prims_log_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::log(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.log.default ge_converter is not implemented!")
