from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nonzero.default)
def conveter_aten_nonzero_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::nonzero(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.nonzero.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.nonzero.out)
def conveter_aten_nonzero_out(
    self: Tensor, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.nonzero.out ge_converter is not implemented!")
