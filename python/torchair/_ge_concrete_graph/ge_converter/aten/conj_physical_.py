from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.conj_physical_.default)
def conveter_aten_conj_physical__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.conj_physical_.default ge_converter is not implemented!")
