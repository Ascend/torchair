from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.complex.default)
def conveter_aten_complex_default(real: Tensor, imag: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::complex(Tensor real, Tensor imag) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.complex.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.complex.out)
def conveter_aten_complex_out(
    real: Tensor, imag: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.complex.out ge_converter is not implemented!")
