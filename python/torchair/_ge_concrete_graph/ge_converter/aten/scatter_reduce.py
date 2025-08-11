from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.scatter_reduce.two)
def conveter_aten_scatter_reduce_two(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str,
    *,
    include_self: bool = True,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter_reduce.two ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_reduce.two_out)
def conveter_aten_scatter_reduce_two_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str,
    *,
    include_self: bool = True,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_reduce.two_out ge_converter is not implemented!")
