from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.scatter_reduce_.two)
def conveter_aten_scatter_reduce__two(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str,
    *,
    include_self: bool = True,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_reduce_.two(Tensor(a!) self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_reduce_.two ge_converter is not implemented!")
