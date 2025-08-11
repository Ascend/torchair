from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.transpose_.default)
def conveter_aten_transpose__default(
    self: Tensor, dim0: int, dim1: int, meta_outputs: TensorSpec = None
):
    """NB: aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.transpose_.default ge_converter is not implemented!")
