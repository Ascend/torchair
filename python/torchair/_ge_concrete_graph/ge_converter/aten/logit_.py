from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.logit_.default)
def conveter_aten_logit__default(
    self: Tensor, eps: Optional[float] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logit_.default ge_converter is not implemented!")
