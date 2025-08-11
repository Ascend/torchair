from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu_factor_ex.default)
def conveter_aten_linalg_lu_factor_ex_default(
    A: Tensor,
    *,
    pivot: bool = True,
    check_errors: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)"""
    raise NotImplementedError("torch.ops.aten.linalg_lu_factor_ex.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_lu_factor_ex.out)
def conveter_aten_linalg_lu_factor_ex_out(
    A: Tensor,
    *,
    pivot: bool = True,
    check_errors: bool = False,
    LU: Tensor = None,
    pivots: Tensor = None,
    info: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_lu_factor_ex.out(Tensor A, *, bool pivot=True, bool check_errors=False, Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info)"""
    raise NotImplementedError("torch.ops.aten.linalg_lu_factor_ex.out ge_converter is not implemented!")
