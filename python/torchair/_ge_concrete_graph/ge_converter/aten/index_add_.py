from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index_add_.default)
def conveter_aten_index_add__default(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_add_.default ge_converter is not implemented!")
