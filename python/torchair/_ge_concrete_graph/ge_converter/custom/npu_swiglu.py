from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2)),
    Support(F16(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_swiglu.default)
def conveter_aten_swiglu_default(
        self: Tensor,
        dim: int = -1,
        meta_outputs: TensorSpec = None):
    """ NB: aten::swiglu(Tensor self) -> Tensor """
    return ge.SwiGlu(self, dim=dim)
