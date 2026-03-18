from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.log_sigmoid_forward.default)
def conveter_aten_log_sigmoid_forward_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)"""
    result = ge.LogSigmoid(self)
    buffer = ge.Identity(self)
    return result, buffer


@register_fx_node_ge_converter(torch.ops.aten.log_sigmoid_forward.output)
def conveter_aten_log_sigmoid_forward_output(
    self: Tensor,
    *,
    output: Tensor = None,
    buffer: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))"""
    res = ge.LogSigmoid(self)
    buf = ge.Identity(self)
    return res, buf
