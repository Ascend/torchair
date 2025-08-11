from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.maximum.default)
def conveter_aten_maximum_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::maximum(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Maximum(self, other)


@register_fx_node_ge_converter(torch.ops.aten.maximum.out)
def conveter_aten_maximum_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.maximum.out ge_converter is not implemented!")
