from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.addmv_.default)
def conveter_aten_addmv__default(
    self: Tensor,
    mat: Tensor,
    vec: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addmv_.default ge_converter is not implemented!")
