from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index_copy.default)
def conveter_aten_index_copy_default(
    self: Tensor, dim: int, index: Tensor, source: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_copy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_copy.dimname)
def conveter_aten_index_copy_dimname(
    self: Tensor, dim: str, index: Tensor, source: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_copy.dimname(Tensor self, str dim, Tensor index, Tensor source) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_copy.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_copy.out)
def conveter_aten_index_copy_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_copy.out(Tensor self, int dim, Tensor index, Tensor source, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_copy.out ge_converter is not implemented!")
