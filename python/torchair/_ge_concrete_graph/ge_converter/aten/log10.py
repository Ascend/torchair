from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(5)),
    Support(F16(5)),
    Support(I64(5)),
    Support(I32(5)),
    Support(I16(5)),
    Support(I8(5)),
])
@register_fx_node_ge_converter(torch.ops.aten.log10.default)
def conveter_aten_log10_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log10(Tensor self) -> Tensor"""
    return ge.Log(self, base=10)


@register_fx_node_ge_converter(torch.ops.aten.log10.out)
def conveter_aten_log10_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log10.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.int)
def conveter_aten_log10_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::log10.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.log10.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.float)
def conveter_aten_log10_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::log10.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.log10.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.complex)
def conveter_aten_log10_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::log10.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.log10.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.Scalar)
def conveter_aten_log10_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::log10.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.log10.Scalar ge_converter is not implemented!")
