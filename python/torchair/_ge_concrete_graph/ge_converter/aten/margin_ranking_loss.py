from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.margin_ranking_loss.default)
def conveter_aten_margin_ranking_loss_default(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0., int reduction=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.margin_ranking_loss.default ge_converter is not implemented!")
