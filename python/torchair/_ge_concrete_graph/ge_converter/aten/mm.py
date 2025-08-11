from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.mm.default)
def conveter_aten_mm_default(self: Tensor, mat2: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::mm(Tensor self, Tensor mat2) -> Tensor"""
    if self.dtype == DataType.DT_INT8 or mat2.dtype == DataType.DT_INT8:
        raise RuntimeError("torch.ops.aten.mm.default ge_converter is not support int8 dtype!")
    return ge.MatMul(self, mat2, None)


@register_fx_node_ge_converter(torch.ops.aten.mm.out)
def conveter_aten_mm_out(
    self: Tensor, mat2: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mm.out ge_converter is not implemented!")
