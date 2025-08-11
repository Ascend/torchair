from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(4, 2)),
        Support(F32(4, 3, 2))
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.view_as_complex.default)
def conveter_aten_view_as_complex_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::view_as_complex(Tensor(a) self) -> Tensor(a)"""
    self = dtype_promote(self, target_dtype=DataType.DT_FLOAT) 
    real = ge.GatherV2(self, 0, [-1], negative_index_support=True)
    imag = ge.GatherV2(self, 1, [-1], negative_index_support=True)
    return ge.Complex(real, imag)
