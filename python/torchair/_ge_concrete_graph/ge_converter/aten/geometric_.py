from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.geometric_.default)
def conveter_aten_geometric__default(
    self: Tensor,
    p: float,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.geometric_.default ge_converter is not implemented!")
