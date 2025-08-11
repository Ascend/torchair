from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_cholesky_ex.default)
def conveter_aten_linalg_cholesky_ex_default(
    self: Tensor,
    *,
    upper: bool = False,
    check_errors: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)"""
    raise NotImplementedError("torch.ops.aten.linalg_cholesky_ex.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_cholesky_ex.L)
def conveter_aten_linalg_cholesky_ex_L(
    self: Tensor,
    *,
    upper: bool = False,
    check_errors: bool = False,
    L: Tensor = None,
    info: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)"""
    raise NotImplementedError("torch.ops.aten.linalg_cholesky_ex.L ge_converter is not implemented!")
