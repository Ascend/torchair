from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.collapse_view.default)
def conveter_prims_collapse_view_default(
    a: Tensor, start: int, end: int, meta_outputs: TensorSpec = None
):
    """NB: prims::collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.collapse_view.default ge_converter is not implemented!")
