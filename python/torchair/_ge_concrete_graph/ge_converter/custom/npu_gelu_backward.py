from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(4, 16, 32), F32(4, 16, 32), approximate="tanh"),
        Support(F32(4, 16, 32), F32(4, 16, 32), approximate="none"),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_gelu_backward.default)
def conveter_aten_npu_gelu_backward(
    grad_output: Tensor,
    self: Tensor,
    *,
    approximate: str = "none",
    meta_outputs: TensorSpec = None
):
    """NB: aten::npu_gelu_backward(Tensor self, *, str approximate="none") -> Tensor"""
    if approximate != "tanh" and approximate != "none": 
        raise ValueError(f"approximate argument must be either none or tanh.")
    return ge.GeluGradV2(grad_output, self, approximate=approximate)
