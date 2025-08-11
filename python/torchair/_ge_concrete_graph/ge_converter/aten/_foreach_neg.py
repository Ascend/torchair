from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._foreach_neg.default)
def conveter_aten__foreach_neg_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_neg(Tensor[] self) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_neg.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_neg.out)
def conveter_aten__foreach_neg_out(
    self: List[Tensor], *, out: List[Tensor] = None
):
    """NB: aten::_foreach_neg.out(Tensor[] self, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_neg.out ge_converter is not implemented!")
