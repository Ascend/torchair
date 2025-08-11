from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._unsafe_index_put.default)
def conveter_aten__unsafe_index_put_default(
    self: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._unsafe_index_put.default ge_converter is not implemented!")
