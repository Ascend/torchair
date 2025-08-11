from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._resize_output_.default)
def conveter_aten__resize_output__default(
    self: Tensor,
    size: Union[List[int], Tensor],
    device: Device,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_resize_output_(Tensor(a!) self, SymInt[] size, Device device) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._resize_output_.default ge_converter is not implemented!")
