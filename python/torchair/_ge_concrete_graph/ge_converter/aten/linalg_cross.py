from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_cross.default)
def conveter_aten_linalg_cross_default(
    self: Tensor, other: Tensor, *, dim: int = -1, meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.linalg_cross.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_cross.out)
def conveter_aten_linalg_cross_out(
    self: Tensor,
    other: Tensor,
    *,
    dim: int = -1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_cross.out(Tensor self, Tensor other, *, int dim=-1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.linalg_cross.out ge_converter is not implemented!")
