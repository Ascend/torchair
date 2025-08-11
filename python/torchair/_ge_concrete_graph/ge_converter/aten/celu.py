from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.celu.default)
def conveter_aten_celu_default(
    self: Tensor, alpha: Union[Number, Tensor] = 1.0, meta_outputs: TensorSpec = None
):
    """NB: aten::celu(Tensor self, Scalar alpha=1.) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.celu.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.celu.out)
def conveter_aten_celu_out(
    self: Tensor,
    alpha: Union[Number, Tensor] = 1.0,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::celu.out(Tensor self, Scalar alpha=1., *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.celu.out ge_converter is not implemented!")
