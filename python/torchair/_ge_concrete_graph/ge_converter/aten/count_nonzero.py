from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.dim_IntList)
def conveter_aten_count_nonzero_dim_IntList(
    self: Tensor, dim: List[int], meta_outputs: TensorSpec = None
):
    """NB: aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.count_nonzero.dim_IntList ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.dim_IntList_out)
def conveter_aten_count_nonzero_dim_IntList_out(
    self: Tensor, dim: List[int], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::count_nonzero.dim_IntList_out(Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.count_nonzero.dim_IntList_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.default)
def conveter_aten_count_nonzero_default(
    self: Tensor, dim: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::count_nonzero(Tensor self, int? dim=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.count_nonzero.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.count_nonzero.out)
def conveter_aten_count_nonzero_out(
    self: Tensor,
    dim: Optional[int] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::count_nonzero.out(Tensor self, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.count_nonzero.out ge_converter is not implemented!")
