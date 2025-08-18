from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(4, 8), approximate="tanh"),
        Support(F32(4, 8), approximate="none"),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_gelu.default)
def conveter_aten_npu_gelu(
    self: Tensor, *, approximate: str = "none", meta_outputs: TensorSpec = None
):
    """NB: aten::npu_gelu(Tensor self, *, str approximate="none") -> Tensor"""
    if approximate != "tanh" and approximate != "none": 
        raise ValueError(f"approximate argument must be either none or tanh.")
    return ge.GeluV2(self, approximate=approximate)
