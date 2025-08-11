from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.conj.default)
def conveter_prims_conj_default(a: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::conj(Tensor(a) a) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.conj.default ge_converter is not implemented!")
