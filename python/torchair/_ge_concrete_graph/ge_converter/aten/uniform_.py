from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.uniform_.default)
def conveter_aten_uniform__default(
    self: Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::uniform_(Tensor(a!) self, float from=0., float to=1., *, Generator? generator=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.uniform_.default ge_converter is not implemented!")
