from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.addmv.default)
def conveter_aten_addmv_default(
    self: Tensor,
    mat: Tensor,
    vec: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.addmv.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.addmv.out)
def conveter_aten_addmv_out(
    self: Tensor,
    mat: Tensor,
    vec: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addmv.out ge_converter is not implemented!")
