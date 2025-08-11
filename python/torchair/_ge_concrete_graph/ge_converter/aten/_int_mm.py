from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._int_mm.out)
def conveter_aten__int_mm_out(
    self: Tensor, mat2: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_int_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._int_mm.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._int_mm.default)
def conveter_aten__int_mm_default(self: Tensor, mat2: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::_int_mm(Tensor self, Tensor mat2) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._int_mm.default ge_converter is not implemented!")
