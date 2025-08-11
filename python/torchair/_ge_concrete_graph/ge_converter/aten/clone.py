from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.clone.default)
def conveter_aten_clone_default(
    self: Tensor, *, memory_format: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""
    if memory_format is not None and memory_format is not torch.contiguous_format:
        raise RuntimeError(
            "torch.ops.aten.clone.default have some unprocessed parameters or cases, "
            "memory_format = {}, torch.contiguous_format = {}".format(memory_format, torch.contiguous_format))

    return ge.Identity(self)


@register_fx_node_ge_converter(torch.ops.aten.clone.out)
def conveter_aten_clone_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.clone.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
