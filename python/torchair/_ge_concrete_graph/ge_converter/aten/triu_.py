from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.triu_.default)
def conveter_aten_triu__default(
    self: Tensor, diagonal: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.triu_.default ge_converter is not implemented!")
