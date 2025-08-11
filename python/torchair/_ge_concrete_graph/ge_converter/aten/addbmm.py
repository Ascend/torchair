from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.addbmm.default)
def conveter_aten_addbmm_default(
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.addbmm.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.addbmm.out)
def conveter_aten_addbmm_out(
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addbmm.out ge_converter is not implemented!")
