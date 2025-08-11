from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._resize_output.default)
def conveter_aten__resize_output_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    device: Device,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_resize_output(Tensor self, SymInt[] size, Device device) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._resize_output.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._resize_output.out)
def conveter_aten__resize_output_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    device: Device,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_resize_output.out(Tensor self, SymInt[] size, Device device, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._resize_output.out ge_converter is not implemented!")
