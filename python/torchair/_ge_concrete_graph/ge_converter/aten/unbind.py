from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(1, 1, 40, 2, 64), 3)
])
@register_fx_node_ge_converter(torch.ops.aten.unbind.int)
def conveter_aten_unbind_int(self: Tensor, dim: int = 0, meta_outputs: List[TensorSpec] = None):
    """NB: aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]"""
    num = len(meta_outputs)
    return ge.Unpack(self, num=num, axis=dim)


@register_fx_node_ge_converter(torch.ops.aten.unbind.Dimname)
def conveter_aten_unbind_Dimname(self: Tensor, dim: str, meta_outputs: List[TensorSpec] = None):
    """NB: aten::unbind.Dimname(Tensor(a -> *) self, str dim) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.unbind.Dimname ge_converter is not implemented!")
