from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(12, 384, 1, value_range=(1, 100))),
        Support(F16(12, 384, value_range=(1, 100))),
        Support(I32(16, 256, value_range=(1, 100))),
        Support(I64(32, 64, value_range=(1, 100))),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.sqrt.default)
def conveter_aten_sqrt_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sqrt(Tensor self) -> Tensor"""
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    return ge.Sqrt(self)


@register_fx_node_ge_converter(torch.ops.aten.sqrt.out)
def conveter_aten_sqrt_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sqrt.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.int)
def conveter_aten_sqrt_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::sqrt.int(int a) -> float"""
    raise RuntimeError("torch.ops.aten.sqrt.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.float)
def conveter_aten_sqrt_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::sqrt.float(float a) -> float"""
    raise RuntimeError("torch.ops.aten.sqrt.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.complex)
def conveter_aten_sqrt_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sqrt.complex(complex a) -> complex"""
    raise RuntimeError("torch.ops.aten.sqrt.complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.Scalar)
def conveter_aten_sqrt_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::sqrt.Scalar(Scalar a) -> Scalar"""
    raise RuntimeError("torch.ops.aten.sqrt.Scalar ge_converter is not supported!")
