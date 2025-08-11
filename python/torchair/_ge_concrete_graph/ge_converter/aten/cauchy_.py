from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.cauchy_.default)
def conveter_aten_cauchy__default(
    self: Tensor,
    median: float = 0.0,
    sigma: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cauchy_(Tensor(a!) self, float median=0., float sigma=1., *, Generator? generator=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cauchy_.default ge_converter is not implemented!")
