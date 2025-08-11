from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.pdist.default)
def conveter_aten_pdist_default(self: Tensor, p: float = 2.0, meta_outputs: TensorSpec = None):
    """NB: aten::pdist(Tensor self, float p=2.) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.pdist.default ge_converter is not implemented!")
