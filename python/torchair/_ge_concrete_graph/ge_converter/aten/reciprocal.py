from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2)),
    Support(F16(3)),
    Support(F16(3, 4, 5)),
    Support(I16(3)),
])
@register_fx_node_ge_converter(torch.ops.aten.reciprocal.default)
def conveter_aten_reciprocal_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::reciprocal(Tensor self) -> Tensor"""
    if self.dtype in [DataType.DT_BOOL, DataType.DT_INT8, DataType.DT_UINT8, DataType.DT_INT16, DataType.DT_INT32,
                    DataType.DT_INT64]:
        return ge.Reciprocal(ge.Cast(self, dst_type=DataType.DT_FLOAT))
    return ge.Reciprocal(self)


@register_fx_node_ge_converter(torch.ops.aten.reciprocal.out)
def conveter_aten_reciprocal_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.reciprocal.out ge_converter is not implemented!")
