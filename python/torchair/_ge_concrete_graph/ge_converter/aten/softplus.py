from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.softplus.default)
def conveter_aten_softplus_default(
    self: Tensor,
    beta: Union[Number, Tensor] = 1,
    threshold: Union[Number, Tensor] = 20,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.softplus.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.softplus.out)
def conveter_aten_softplus_out(
    self: Tensor,
    beta: Union[Number, Tensor] = 1,
    threshold: Union[Number, Tensor] = 20,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.softplus.out ge_converter is not implemented!")
