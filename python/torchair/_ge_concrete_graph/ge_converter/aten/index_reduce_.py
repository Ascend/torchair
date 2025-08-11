from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index_reduce_.default)
def conveter_aten_index_reduce__default(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
    reduce: str,
    *,
    include_self: bool = True,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_reduce_(Tensor(a!) self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_reduce_.default ge_converter is not implemented!")
