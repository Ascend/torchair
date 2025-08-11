from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4, 5)),
    Support(F16(3, 4, 5)),
])
@register_fx_node_ge_converter(torch.ops.aten.sin.default)
def conveter_aten_sin_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sin(Tensor self) -> Tensor"""
    return ge.Sin(self)


@register_fx_node_ge_converter(torch.ops.aten.sin.out)
def conveter_aten_sin_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sin.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.int)
def conveter_aten_sin_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::sin.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.sin.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.float)
def conveter_aten_sin_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::sin.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.sin.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.complex)
def conveter_aten_sin_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sin.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.sin.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.Scalar)
def conveter_aten_sin_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::sin.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.sin.Scalar ge_converter is not implemented!")
