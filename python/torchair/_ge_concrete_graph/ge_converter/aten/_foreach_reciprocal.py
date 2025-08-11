from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._foreach_reciprocal.default)
def conveter_aten__foreach_reciprocal_default(
    self: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_reciprocal(Tensor[] self) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_reciprocal.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_reciprocal.out)
def conveter_aten__foreach_reciprocal_out(
    self: List[Tensor], *, out: List[Tensor] = None
):
    """NB: aten::_foreach_reciprocal.out(Tensor[] self, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_reciprocal.out ge_converter is not implemented!")
