from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.erfc.default)
def conveter_aten_erfc_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::erfc(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.erfc.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.out)
def conveter_aten_erfc_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.erfc.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.int)
def conveter_aten_erfc_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::erfc.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.erfc.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.float)
def conveter_aten_erfc_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::erfc.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.erfc.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.Scalar)
def conveter_aten_erfc_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::erfc.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.erfc.Scalar ge_converter is not implemented!")
