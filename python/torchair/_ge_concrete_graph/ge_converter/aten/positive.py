from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.positive.default)
def conveter_aten_positive_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::positive(Tensor(a) self) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.positive.default ge_converter is not implemented!")
