from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 6, 7, 7), F32(2, 6, 7, 7))
])
@register_fx_node_ge_converter(torch.ops.aten.hardswish_backward.default)
def conveter_aten_hardswish_backward_default(
    grad_output: Tensor, self: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor"""
    return ge.HardSwishGrad(grad_output, self)


@register_fx_node_ge_converter(torch.ops.aten.hardswish_backward.out)
def conveter_aten_hardswish_backward_out(
    grad_output: Tensor, self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::hardswish_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.hardswish_backward.out ge_converter is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
