from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 10))
])
@register_fx_node_ge_converter(torch.ops.aten.exp.default)
def conveter_aten_exp_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::exp(Tensor self) -> Tensor"""
    if self.dtype == DataType.DT_BOOL:
        self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    return ge.Exp(self)


@register_fx_node_ge_converter(torch.ops.aten.exp.out)
def conveter_aten_exp_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.exp.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.exp.int)
def conveter_aten_exp_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::exp.int(int a) -> float"""
    raise RuntimeError("torch.ops.aten.exp.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.exp.float)
def conveter_aten_exp_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::exp.float(float a) -> float"""
    raise RuntimeError("torch.ops.aten.exp.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.exp.complex)
def conveter_aten_exp_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::exp.complex(complex a) -> complex"""
    raise RuntimeError("torch.ops.aten.exp.complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.exp.Scalar)
def conveter_aten_exp_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::exp.Scalar(Scalar a) -> Scalar"""
    raise RuntimeError("torch.ops.aten.exp.Scalar ge_converter is not supported!")
