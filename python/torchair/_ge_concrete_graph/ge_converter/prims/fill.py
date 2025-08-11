from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.fill.default)
def conveter_prims_fill_default(
    self: Tensor, value: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: prims::fill(Tensor self, Scalar value) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.fill.default ge_converter is not implemented!")
