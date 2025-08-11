from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.exponential_.default)
def conveter_aten_exponential__default(
    self: Tensor,
    lambd: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::exponential_(Tensor(a!) self, float lambd=1., *, Generator? generator=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.exponential_.default ge_converter is not implemented!")
