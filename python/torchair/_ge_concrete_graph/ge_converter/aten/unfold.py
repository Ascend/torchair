from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.unfold.default)
def conveter_aten_unfold_default(
    self: Tensor, dimension: int, size: int, step: int, meta_outputs: TensorSpec = None
):
    """NB: aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.unfold.default ge_converter is not implemented!")
