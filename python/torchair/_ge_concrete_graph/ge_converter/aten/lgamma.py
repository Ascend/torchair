from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.lgamma.default)
def conveter_aten_lgamma_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::lgamma(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.lgamma.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.out)
def conveter_aten_lgamma_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lgamma.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.int)
def conveter_aten_lgamma_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::lgamma.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.lgamma.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.float)
def conveter_aten_lgamma_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::lgamma.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.lgamma.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.Scalar)
def conveter_aten_lgamma_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::lgamma.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.lgamma.Scalar ge_converter is not implemented!")
