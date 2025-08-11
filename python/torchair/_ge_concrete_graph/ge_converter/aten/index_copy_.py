from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index_copy_.default)
def conveter_aten_index_copy__default(
    self: Tensor, dim: int, index: Tensor, source: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_copy_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_copy_.dimname)
def conveter_aten_index_copy__dimname(
    self: Tensor, dim: str, index: Tensor, source: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_copy_.dimname(Tensor(a!) self, str dim, Tensor index, Tensor source) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_copy_.dimname ge_converter is not implemented!")
