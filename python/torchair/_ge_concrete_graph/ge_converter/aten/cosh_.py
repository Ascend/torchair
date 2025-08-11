from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.cosh_.default)
def conveter_aten_cosh__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::cosh_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cosh_.default ge_converter is not implemented!")
