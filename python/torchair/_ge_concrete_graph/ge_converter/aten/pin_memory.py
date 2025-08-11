from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.pin_memory.default)
def conveter_aten_pin_memory_default(
    self: Tensor, device: Optional[Device] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.pin_memory.default ge_converter is not implemented!")
