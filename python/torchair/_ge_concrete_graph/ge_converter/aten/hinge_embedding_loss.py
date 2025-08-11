from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.hinge_embedding_loss.default)
def conveter_aten_hinge_embedding_loss_default(
    self: Tensor,
    target: Tensor,
    margin: float = 1.0,
    reduction: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1., int reduction=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.hinge_embedding_loss.default ge_converter is not implemented!")
