from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nll_loss2d_forward.default)
def conveter_aten_nll_loss2d_forward_default(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)"""
    raise NotImplementedError("torch.ops.aten.nll_loss2d_forward.default ge_converter is not implemented!")