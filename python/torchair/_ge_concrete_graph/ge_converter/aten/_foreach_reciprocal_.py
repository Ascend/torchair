from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._foreach_reciprocal_.default)
def conveter_aten__foreach_reciprocal__default(
    self: List[Tensor]
):
    """NB: aten::_foreach_reciprocal_(Tensor(a!)[] self) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_reciprocal_.default ge_converter is not implemented!")
