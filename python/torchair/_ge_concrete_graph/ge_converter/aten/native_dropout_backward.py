from torchair._ge_concrete_graph.ge_converter.converter_utils import *


# No testcase because the dtype and shape of input *mask* are different from cpu's.
@register_fx_node_ge_converter(torch.ops.aten.native_dropout_backward.default)
def conveter_aten_native_dropout_backward_default(
    grad_output: Tensor, mask: Tensor, scale: float, meta_outputs: TensorSpec = None
):
    """NB: aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor"""
    retain = 1. / scale
    return ge.DropOutDoMask(grad_output, mask, retain)


@register_fx_node_ge_converter(torch.ops.aten.native_dropout_backward.out)
def conveter_aten_native_dropout_backward_out(
    grad_output: Tensor,
    mask: Tensor,
    scale: float,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_dropout_backward.out(Tensor grad_output, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.native_dropout_backward.out ge_converter is not implemented!")
