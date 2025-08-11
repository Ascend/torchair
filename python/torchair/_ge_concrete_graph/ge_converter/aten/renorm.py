from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.renorm.default)
def conveter_aten_renorm_default(
    self: Tensor,
    p: Union[Number, Tensor],
    dim: int,
    maxnorm: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.renorm.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.renorm.out)
def conveter_aten_renorm_out(
    self: Tensor,
    p: Union[Number, Tensor],
    dim: int,
    maxnorm: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.renorm.out ge_converter is not implemented!")
