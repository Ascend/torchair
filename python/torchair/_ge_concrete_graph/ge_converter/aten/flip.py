from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4, 5), [0]),
    Support(I32(3, 4), [1]),
    Support(I32(3, 4), [0, 1]),
])
@register_fx_node_ge_converter(torch.ops.aten.flip.default)
def conveter_aten_flip_default(self: Tensor, dims: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::flip(Tensor self, int[] dims) -> Tensor"""
    return ge.ReverseV2(self, dims)


@register_fx_node_ge_converter(torch.ops.aten.flip.out)
def conveter_aten_flip_out(
    self: Tensor, dims: List[int], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::flip.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.flip.out ge_converter is not implemented!")
