from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.device_put.default)
def conveter_prims_device_put_default(
    a: Tensor, device: Device, meta_outputs: TensorSpec = None
):
    """NB: prims::device_put(Tensor a, Device device) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.device_put.default ge_converter is not implemented!")
