from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.negative_.default)
def conveter_aten_negative__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::negative_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.negative_.default ge_converter is not implemented!")
