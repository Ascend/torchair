from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.shift_right_arithmetic.default)
def conveter_prims_shift_right_arithmetic_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::shift_right_arithmetic(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.shift_right_arithmetic.default ge_converter is not implemented!")
