from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.normal_functional.default)
def conveter_aten_normal_functional_default(
    self: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::normal_functional(Tensor self, float mean=0., float std=1., *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.normal_functional.default ge_converter is not implemented!")
