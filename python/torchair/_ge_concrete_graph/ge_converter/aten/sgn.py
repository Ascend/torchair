from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.sgn.default)
def conveter_aten_sgn_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sgn(Tensor self) -> Tensor"""
    # The complex data type is not supported, 28 represent the unknown dtype!
    if self.dtype == 28:
        raise NotImplementedError("torch.ops.aten.sgn.tensor ge_converter with input of complex is not implemented!")
    return ge.Sign(self)


@register_fx_node_ge_converter(torch.ops.aten.sgn.out)
def conveter_aten_sgn_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sgn.out ge_converter is not supported!")
