from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(BOOL(2, 2)),
    Support(I16(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.logical_not.default)
def conveter_aten_logical_not_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::logical_not(Tensor self) -> Tensor"""
    self = dtype_promote(self, target_dtype=DataType.DT_BOOL)
    result_logicalnot = ge.LogicalNot(self)
    if meta_outputs and meta_outputs.dtype != DataType.DT_BOOL:
        result_logicalnot = dtype_promote(result_logicalnot, target_dtype=meta_outputs.dtype)
    return result_logicalnot


@register_fx_node_ge_converter(torch.ops.aten.logical_not.out)
def conveter_aten_logical_not_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logical_not.out ge_converter is not implemented!")
