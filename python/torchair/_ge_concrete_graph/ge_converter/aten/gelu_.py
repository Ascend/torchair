from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.gelu_.default)
def conveter_aten_gelu__default(
    self: Tensor, *, approximate: str = "None", meta_outputs: TensorSpec = None
):
    """NB: aten::gelu_(Tensor(a!) self, *, str approximate="none") -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gelu_.default ge_converter is not implemented!")
