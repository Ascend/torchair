from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.resize_as_.default)
def conveter_aten_resize_as__default(
    self: Tensor,
    the_template: Tensor,
    *,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.resize_as_.default ge_converter is not implemented!")
