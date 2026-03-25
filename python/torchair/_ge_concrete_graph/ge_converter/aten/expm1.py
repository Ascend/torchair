from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.expm1.default)
def conveter_aten_expm1_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::expm1(Tensor self) -> Tensor"""
    return ge.Expm1(self)


@register_fx_node_ge_converter(torch.ops.aten.expm1.out)
def conveter_aten_expm1_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    out = ge.Expm1(self)
    return out


@register_fx_node_ge_converter(torch.ops.aten.expm1.int)
def conveter_aten_expm1_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::expm1.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.expm1.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.float)
def conveter_aten_expm1_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::expm1.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.expm1.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.Scalar)
def conveter_aten_expm1_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::expm1.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.expm1.Scalar ge_converter is not implemented!")
