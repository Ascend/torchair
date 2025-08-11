from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.isinf.default)
def conveter_aten_isinf_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::isinf(Tensor self) -> Tensor"""
    return ge.IsInf(self)


@register_fx_node_ge_converter(torch.ops.aten.isinf.out)
def conveter_aten_isinf_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::isinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.isinf.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.isinf.float)
def conveter_aten_isinf_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::isinf.float(float a) -> bool"""
    raise NotImplementedError("torch.ops.aten.isinf.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.isinf.complex)
def conveter_aten_isinf_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::isinf.complex(complex a) -> bool"""
    raise NotImplementedError("torch.ops.aten.isinf.complex ge_converter is not implemented!")
