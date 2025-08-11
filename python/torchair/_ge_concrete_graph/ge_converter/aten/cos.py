from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4, 5)),
    Support(F16(3, 4, 5)),
])
@register_fx_node_ge_converter(torch.ops.aten.cos.default)
def conveter_aten_cos_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::cos(Tensor self) -> Tensor"""
    return ge.Cos(self)


@register_fx_node_ge_converter(torch.ops.aten.cos.out)
def conveter_aten_cos_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cos.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.int)
def conveter_aten_cos_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::cos.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.cos.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.float)
def conveter_aten_cos_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::cos.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.cos.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.complex)
def conveter_aten_cos_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::cos.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.cos.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.Scalar)
def conveter_aten_cos_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::cos.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.cos.Scalar ge_converter is not implemented!")
