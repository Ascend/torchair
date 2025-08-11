from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4, 5)),
    Support(F16(3, 4, 5)),
])
@register_fx_node_ge_converter(torch.ops.aten.sinh.default)
def conveter_aten_sinh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sinh(Tensor self) -> Tensor"""
    return ge.Sinh(self)


@register_fx_node_ge_converter(torch.ops.aten.sinh.out)
def conveter_aten_sinh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sinh.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.int)
def conveter_aten_sinh_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::sinh.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.sinh.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.float)
def conveter_aten_sinh_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::sinh.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.sinh.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.complex)
def conveter_aten_sinh_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sinh.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.sinh.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.Scalar)
def conveter_aten_sinh_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::sinh.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.sinh.Scalar ge_converter is not implemented!")
