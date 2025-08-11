from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.angle.default)
def conveter_aten_angle_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::angle(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.angle.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.angle.out)
def conveter_aten_angle_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.angle.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.angle.int)
def conveter_aten_angle_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::angle.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.angle.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.angle.float)
def conveter_aten_angle_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::angle.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.angle.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.angle.complex)
def conveter_aten_angle_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::angle.complex(complex a) -> float"""
    raise NotImplementedError("torch.ops.aten.angle.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.angle.Scalar)
def conveter_aten_angle_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::angle.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.angle.Scalar ge_converter is not implemented!")
