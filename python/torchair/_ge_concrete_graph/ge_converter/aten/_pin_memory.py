from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._pin_memory.default)
def conveter_aten__pin_memory_default(
    self: Tensor, device: Optional[Device] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_pin_memory(Tensor self, Device? device=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._pin_memory.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._pin_memory.out)
def conveter_aten__pin_memory_out(
    self: Tensor,
    device: Optional[Device] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_pin_memory.out(Tensor self, Device? device=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._pin_memory.out ge_converter is not implemented!")
