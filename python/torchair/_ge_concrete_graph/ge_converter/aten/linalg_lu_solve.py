from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu_solve.default)
def conveter_aten_linalg_lu_solve_default(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.linalg_lu_solve.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu_solve.out)
def conveter_aten_linalg_lu_solve_out(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu_solve.out(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.linalg_lu_solve.out ge_converter is not implemented!")
