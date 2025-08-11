from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.logit.default)
def conveter_aten_logit_default(
    self: Tensor, eps: Optional[float] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logit(Tensor self, float? eps=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logit.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logit.out)
def conveter_aten_logit_out(
    self: Tensor,
    eps: Optional[float] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logit.out ge_converter is not implemented!")
