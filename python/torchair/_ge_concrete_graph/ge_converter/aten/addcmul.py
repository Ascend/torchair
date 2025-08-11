from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(1024, 1024), F32(1024, 1024), F32(1024, 1024)),
        Support(F32(1024, 1024), F32(1024, 1024), F32(1024, 1024), value=-1),
        Support(F32(1024, 1024), F32(1024, 1024), F32(1024, 1024), value=0.1),
        Support(F16(1024, 1024), F16(1024, 1024), F16(1024, 1024), value=0.1),
        Support(F32(1024, 1024), F16(1024, 1024), F16(1024, 1024), value=0.1),
        Support(F16(1024, 1024), F32(1024, 1024), F32(1024, 1024), value=0.1),
        Support(BF16(1024, 1024), BF16(1024, 1024), BF16(1024, 1024), value=0.1),
        Support(F32(1024), F32(1024), F32(1024))
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.addcmul.default)
def conveter_aten_addcmul_default(
    self: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor"""
    self, tensor1, tensor2, value = dtype_promote(self, tensor1, tensor2, value, target_dtype=meta_outputs.dtype)
    return ge.Addcmul(self, tensor1, tensor2, value)


@register_fx_node_ge_converter(torch.ops.aten.addcmul.out)
def conveter_aten_addcmul_out(
    self: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addcmul.out ge_converter is not implemented!")
