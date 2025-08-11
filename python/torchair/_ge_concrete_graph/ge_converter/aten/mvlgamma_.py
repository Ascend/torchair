from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.mvlgamma_.default)
def conveter_aten_mvlgamma__default(self: Tensor, p: int, meta_outputs: TensorSpec = None):
    """NB: aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mvlgamma_.default ge_converter is not implemented!")
