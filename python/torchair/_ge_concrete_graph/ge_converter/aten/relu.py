from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2)),
    Support(F16(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.relu.default)
def conveter_aten_relu_default(
        self: Tensor,
        meta_outputs: TensorSpec = None):
    """ NB: aten::relu(Tensor self) -> Tensor """
    return ge.Relu(self)


@register_fx_node_ge_converter(torch.ops.aten.relu.out)
def conveter_aten_relu_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.relu.out ge_converter is not implemented!")
