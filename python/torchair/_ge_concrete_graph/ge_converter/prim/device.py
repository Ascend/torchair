from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prim.device.default)
def conveter_prim_device_default(a: Tensor, meta_outputs: TensorSpec = None):
    """NB: prim::device(Tensor a) -> Device"""
    raise NotImplementedError("torch.ops.prim.device.default ge_converter is not implemented!")
