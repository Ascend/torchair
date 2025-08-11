from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.smooth_l1_loss.default)
def conveter_aten_smooth_l1_loss_default(
    self: Tensor,
    target: Tensor,
    reduction: int = 1,
    beta: float = 1.0,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=1, float beta=1.) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.smooth_l1_loss.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.smooth_l1_loss.out)
def conveter_aten_smooth_l1_loss_out(
    self: Tensor,
    target: Tensor,
    reduction: int = 1,
    beta: float = 1.0,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=1, float beta=1., *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.smooth_l1_loss.out ge_converter is not implemented!")
