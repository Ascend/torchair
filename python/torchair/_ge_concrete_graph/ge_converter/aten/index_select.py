from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4), 1, I32(2, value_range=(0, 4))),
    Support(F32(3, 4), -1, I32(2, value_range=(0, 4))),
    Support(F32(3, 10), 1, I32(2, value_range=(0, 10))),
])
@register_fx_node_ge_converter(torch.ops.aten.index_select.default)
def conveter_aten_index_select_default(
    self: Tensor, dim: int, index: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_select(Tensor self, int dim, Tensor index) -> Tensor"""
    dim = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    return ge.GatherV2(self, index, dim)


@register_fx_node_ge_converter(torch.ops.aten.index_select.out)
def conveter_aten_index_select_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.index_select.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.index_select.dimname)
def conveter_aten_index_select_dimname(
    self: Tensor, dim: str, index: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_select.dimname(Tensor self, str dim, Tensor index) -> Tensor"""
    raise RuntimeError("torch.ops.aten.index_select.dimname is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.index_select.dimname_out)
def conveter_aten_index_select_dimname_out(
    self: Tensor,
    dim: str,
    index: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_select.dimname_out(Tensor self, str dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.index_select.dimname_out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
