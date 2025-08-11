from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 2, 2), F32(2, 2, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.bmm.default)
def conveter_aten_bmm_default(self: Tensor, mat2: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::bmm(Tensor self, Tensor mat2) -> Tensor"""
    if self.dtype == DataType.DT_INT8 or mat2.dtype == DataType.DT_INT8:
        raise RuntimeError("torch.ops.aten.bmm.default ge_converter is not support int8 dtype!")
    self, mat2 = dtype_promote(self, mat2, target_dtype=meta_outputs.dtype)
    return ge.BatchMatMul(self, mat2)


@register_fx_node_ge_converter(torch.ops.aten.bmm.out)
def conveter_aten_bmm_out(
    self: Tensor, mat2: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bmm.out ge_converter is not implemented!")
