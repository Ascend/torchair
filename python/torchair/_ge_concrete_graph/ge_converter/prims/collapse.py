from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.collapse.default)
def conveter_prims_collapse_default(
    a: Tensor, start: int, end: int, meta_outputs: TensorSpec = None
):
    """NB: prims::collapse(Tensor a, int start, int end) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.collapse.default ge_converter is not implemented!")
