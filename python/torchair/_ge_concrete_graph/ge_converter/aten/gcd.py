from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.gcd.default)
def conveter_aten_gcd_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::gcd(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.gcd.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gcd.out)
def conveter_aten_gcd_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gcd.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gcd.int)
def conveter_aten_gcd_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::gcd.int(int a, int b) -> int"""
    raise NotImplementedError("torch.ops.aten.gcd.int ge_converter is not implemented!")
