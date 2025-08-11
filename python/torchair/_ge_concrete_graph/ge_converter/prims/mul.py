from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.mul.default)
def conveter_prims_mul_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::mul(Tensor self, Tensor other) -> Tensor"""
    return ge.Mul(self, other)
