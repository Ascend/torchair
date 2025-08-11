from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.tril_.default)
def conveter_aten_tril__default(
    self: Tensor, diagonal: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.tril_.default ge_converter is not implemented!")
