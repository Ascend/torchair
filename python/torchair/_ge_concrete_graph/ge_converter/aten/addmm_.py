from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.addmm_.default)
def conveter_aten_addmm__default(
    self: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addmm_.default ge_converter is not implemented!")
