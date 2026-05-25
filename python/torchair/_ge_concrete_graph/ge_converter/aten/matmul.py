from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 3), F32(3, 4)),
        Support(F32(10, 2, 3), F32(10, 3, 4)),
        Support(F16(3, 2, 2), F16(2, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.matmul.default)
def conveter_aten_matmul_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::matmul(Tensor self, Tensor other) -> Tensor"""
    if self.dtype == DataType.DT_INT8 or other.dtype == DataType.DT_INT8:
        raise RuntimeError("torch.ops.aten.matmul.default ge_converter is not support int8 dtype!")
    if self.rank >= 3 or other.rank >= 3:
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
        return ge.BatchMatMul(self, other)
    return ge.MatMul(self, other, None)


@register_fx_node_ge_converter(torch.ops.aten.matmul.out)
def conveter_aten_matmul_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.matmul.out ge_converter is not implemented!")
