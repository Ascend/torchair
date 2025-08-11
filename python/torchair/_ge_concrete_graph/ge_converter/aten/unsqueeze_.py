from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.unsqueeze_.default)
def conveter_aten_unsqueeze__default(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.unsqueeze_.default ge_converter is not implemented!")
