from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(5)),
    Support(F16(5))
])
@register_fx_node_ge_converter(torch.ops.aten.log1p.default)
def conveter_aten_log1p_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log1p(Tensor self) -> Tensor"""
    return ge.Log1p(self)


@register_fx_node_ge_converter(torch.ops.aten.log1p.out)
def conveter_aten_log1p_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log1p.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log1p.int)
def conveter_aten_log1p_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::log1p.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.log1p.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log1p.float)
def conveter_aten_log1p_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::log1p.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.log1p.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log1p.Scalar)
def conveter_aten_log1p_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::log1p.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.log1p.Scalar ge_converter is not implemented!")
