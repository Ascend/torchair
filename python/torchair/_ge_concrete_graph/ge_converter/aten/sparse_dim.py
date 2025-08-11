from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.sparse_dim.default)
def conveter_aten_sparse_dim_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sparse_dim(Tensor self) -> int"""
    raise NotImplementedError("torch.ops.aten.sparse_dim.default ge_converter is not implemented!")
