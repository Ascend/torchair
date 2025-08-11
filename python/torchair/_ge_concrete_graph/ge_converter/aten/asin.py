from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(1024)),
    Support(F16(1024)),
    Support(F32(10, 40, 1024)),
    Support(F16(10, 40, 1024)),
])
@register_fx_node_ge_converter(torch.ops.aten.asin.default)
def conveter_aten_asin_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::asin(Tensor self) -> Tensor"""
    if meta_outputs:
        self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    return ge.Asin(self)


@register_fx_node_ge_converter(torch.ops.aten.asin.out)
def conveter_aten_asin_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.asin.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.int)
def conveter_aten_asin_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::asin.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.asin.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.float)
def conveter_aten_asin_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::asin.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.asin.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.complex)
def conveter_aten_asin_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::asin.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.asin.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.Scalar)
def conveter_aten_asin_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::asin.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.asin.Scalar ge_converter is not implemented!")
