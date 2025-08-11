from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(1, 1, 40, 128)),
    Support(U8(1, 1, 40, 128))
])
@register_fx_node_ge_converter(torch.ops.aten.alias.default)
def conveter_aten_alias_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::alias(Tensor(a) self) -> Tensor(a)"""
    shape = ge.Shape(self)
    shape = dtype_promote(shape, target_dtype=DataType.DT_INT64)
    return ge.Reshape(self, shape)
