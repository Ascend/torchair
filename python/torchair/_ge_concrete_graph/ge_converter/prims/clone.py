from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.clone.default)
def conveter_prims_clone_default(
    self: Tensor, *, memory_format: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: prims::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.clone.default ge_converter is not implemented!")
