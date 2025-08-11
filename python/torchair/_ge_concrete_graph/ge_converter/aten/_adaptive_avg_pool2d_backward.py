from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool2d_backward.default)
def conveter_aten__adaptive_avg_pool2d_backward_default(
    grad_output: Tensor, self: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor"""
    if self.symsize is not None and all([not isinstance(s, torch.SymInt) for s in self.symsize]):
        return ge.AdaptiveAvgPool2dGrad(input_grad=grad_output, orig_input_shape=self.symsize)
    raise NotImplementedError("torch.ops.aten._adaptive_avg_pool2d_backward.default ge_converter is not implemented "
                              "when self is dynamic")


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool2d_backward.out)
def conveter_aten__adaptive_avg_pool2d_backward_out(
    grad_output: Tensor, self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool2d_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten._adaptive_avg_pool2d_backward.out ge_converter is not supported!")
