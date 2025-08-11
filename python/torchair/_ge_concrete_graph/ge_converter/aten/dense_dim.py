from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.dense_dim.default)
def conveter_aten_dense_dim_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::dense_dim(Tensor self) -> int"""
    raise NotImplementedError("torch.ops.aten.dense_dim.default ge_converter is not implemented!")
