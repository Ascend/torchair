from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index_put_.default)
def conveter_aten_index_put__default(
    self: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_put_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_put_.hacked_twin)
def conveter_aten_index_put__hacked_twin(
    self: Tensor,
    indices: List[Tensor],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_put_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_put_.hacked_twin ge_converter is not implemented!")
