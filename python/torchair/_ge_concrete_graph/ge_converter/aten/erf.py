from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 3)),
])
@register_fx_node_ge_converter(torch.ops.aten.erf.default)
def conveter_aten_erf_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::erf(Tensor self) -> Tensor"""
    return ge.Erf(self)


@register_fx_node_ge_converter(torch.ops.aten.erf.out)
def conveter_aten_erf_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.erf.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.erf.int)
def conveter_aten_erf_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::erf.int(int a) -> float"""
    raise RuntimeError("torch.ops.aten.erf.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.erf.float)
def conveter_aten_erf_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::erf.float(float a) -> float"""
    raise RuntimeError("torch.ops.aten.erf.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.erf.Scalar)
def conveter_aten_erf_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::erf.Scalar(Scalar a) -> Scalar"""
    raise RuntimeError("torch.ops.aten.erf.Scalar ge_converter is not supported!")
