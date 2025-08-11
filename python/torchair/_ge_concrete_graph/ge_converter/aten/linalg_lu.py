from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu.default)
def conveter_aten_linalg_lu_default(
    A: Tensor, *, pivot: bool = True, meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)"""
    raise NotImplementedError("torch.ops.aten.linalg_lu.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu.out)
def conveter_aten_linalg_lu_out(
    A: Tensor,
    *,
    pivot: bool = True,
    P: Tensor = None,
    L: Tensor = None,
    U: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)"""
    raise NotImplementedError("torch.ops.aten.linalg_lu.out ge_converter is not implemented!")
