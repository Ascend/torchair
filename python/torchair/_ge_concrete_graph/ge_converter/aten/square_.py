from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.square_.default)
def conveter_aten_square__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::square_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.square_.default ge_converter is not implemented!")
