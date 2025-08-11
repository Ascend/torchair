from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(1024)),
    Support(F16(1024)),
    Support(F32(10, 40, 1024)),
    Support(F16(10, 40, 1024)),
])
@register_fx_node_ge_converter(torch.ops.aten.atanh.default)
def conveter_aten_atanh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::atanh(Tensor self) -> Tensor"""
    return ge.Atanh(self)


@register_fx_node_ge_converter(torch.ops.aten.atanh.out)
def conveter_aten_atanh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.atanh.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.int)
def conveter_aten_atanh_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::atanh.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.atanh.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.float)
def conveter_aten_atanh_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::atanh.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.atanh.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.complex)
def conveter_aten_atanh_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::atanh.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.atanh.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.Scalar)
def conveter_aten_atanh_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::atanh.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.atanh.Scalar ge_converter is not implemented!")
