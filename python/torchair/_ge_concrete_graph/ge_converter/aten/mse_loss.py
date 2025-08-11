from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def get_reduction_str(reduction):
    reduction_str = ["none", "mean", "sum"]
    return reduction_str[reduction]


@declare_supported([
    Support(F32(6, 512, 44, 44), F32(6, 512, 44, 44), 0),
    Support(F32(6, 512, 44, 44), F32(6, 512, 44, 44), 1),
    Support(F32(6, 512, 44, 44), F32(6, 512, 44, 44), 2),
])
@register_fx_node_ge_converter(torch.ops.aten.mse_loss.default)
def conveter_aten_mse_loss_default(
    self: Tensor, target: Tensor, reduction: int = 1, meta_outputs: TensorSpec = None
):
    """NB: aten::mse_loss(Tensor self, Tensor target, int reduction=1) -> Tensor"""
    self, target = dtype_promote(self, target, target_dtype=meta_outputs.dtype)
    reduction_str = get_reduction_str(reduction)
    return ge.MseLoss(self, target, reduction=reduction_str)


@register_fx_node_ge_converter(torch.ops.aten.mse_loss.out)
def conveter_aten_mse_loss_out(
    self: Tensor,
    target: Tensor,
    reduction: int = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::mse_loss.out(Tensor self, Tensor target, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.mse_loss.out ge_converter is not supported!")
