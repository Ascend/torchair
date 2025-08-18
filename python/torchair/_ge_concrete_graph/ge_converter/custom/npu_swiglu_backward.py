from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
    Support(F16(2, 2), F16(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_swiglu_backward.default)
def conveter_aten_swiglu_backward_default(
        y_grad: Tensor,
        x: Tensor,
        dim: int = -1,
        meta_outputs: TensorSpec = None):
    """ NB: aten::swiglu_backward(Tensor y_grad, Tensor self) -> Tensor """
    return ge.SwiGluGrad(y_grad, x, dim=dim)
