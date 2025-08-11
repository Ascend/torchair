from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._local_scalar_dense.default)
def conveter_aten__local_scalar_dense_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::_local_scalar_dense(Tensor self) -> Scalar"""
    raise NotImplementedError("torch.ops.aten._local_scalar_dense.default ge_converter is not implemented!")
