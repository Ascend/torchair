from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._foreach_asin.default)
def conveter_aten__foreach_asin_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    return ge.ForeachASin(x=self)
