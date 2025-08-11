from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.rad2deg_.default)
def conveter_aten_rad2deg__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rad2deg_.default ge_converter is not implemented!")
