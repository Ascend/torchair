from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_solve.default)
def conveter_aten_linalg_ldl_solve_default(
    LD: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    hermitian: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.linalg_ldl_solve.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_solve.out)
def conveter_aten_linalg_ldl_solve_out(
    LD: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    hermitian: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_ldl_solve.out(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.linalg_ldl_solve.out ge_converter is not implemented!")
