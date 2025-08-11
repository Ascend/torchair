from torchair._ge_concrete_graph.ge_converter.converter_utils import *
 

@declare_supported(
    [
        Support(F32(2, 6, 1, 1)),
        Support(F32(96, 65)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.sigmoid.default)
def conveter_aten_sigmoid_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sigmoid(Tensor self) -> Tensor"""
    return ge.Sigmoid(self)


@register_fx_node_ge_converter(torch.ops.aten.sigmoid.out)
def conveter_aten_sigmoid_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sigmoid.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
