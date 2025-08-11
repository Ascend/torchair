from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.where.default)
def conveter_prims_where_default(
    pred: Tensor, a: Tensor, b: Tensor, meta_outputs: TensorSpec = None
):
    """NB: prims::where(Tensor pred, Tensor a, Tensor b) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.where.default ge_converter is not implemented!")
