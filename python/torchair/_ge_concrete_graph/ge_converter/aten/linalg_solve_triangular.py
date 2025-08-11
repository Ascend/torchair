from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_solve_triangular.default)
def conveter_aten_linalg_solve_triangular_default(
    self: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.linalg_solve_triangular.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_solve_triangular.out)
def conveter_aten_linalg_solve_triangular_out(
    self: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_solve_triangular.out(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.linalg_solve_triangular.out ge_converter is not implemented!")
