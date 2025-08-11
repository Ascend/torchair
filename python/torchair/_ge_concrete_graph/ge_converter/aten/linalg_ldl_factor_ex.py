from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_factor_ex.default)
def conveter_aten_linalg_ldl_factor_ex_default(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)"""
    raise NotImplementedError("torch.ops.aten.linalg_ldl_factor_ex.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_factor_ex.out)
def conveter_aten_linalg_ldl_factor_ex_out(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
    LD: Tensor = None,
    pivots: Tensor = None,
    info: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info)"""
    raise NotImplementedError("torch.ops.aten.linalg_ldl_factor_ex.out ge_converter is not implemented!")
