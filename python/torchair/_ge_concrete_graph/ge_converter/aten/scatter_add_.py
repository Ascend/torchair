from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.scatter_add_.default)
def conveter_aten_scatter_add__default(
    self: Tensor, dim: int, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_add_.default ge_converter is not implemented!")
