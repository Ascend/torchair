from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.isnan.default)
def conveter_aten_isnan_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::isnan(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.isnan.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.isnan.out)
def conveter_aten_isnan_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::isnan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.isnan.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.isnan.float)
def conveter_aten_isnan_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::isnan.float(float a) -> bool"""
    raise NotImplementedError("torch.ops.aten.isnan.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.isnan.complex)
def conveter_aten_isnan_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::isnan.complex(complex a) -> bool"""
    raise NotImplementedError("torch.ops.aten.isnan.complex ge_converter is not implemented!")
