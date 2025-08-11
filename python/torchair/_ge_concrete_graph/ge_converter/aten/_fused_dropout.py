from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._fused_dropout.default)
def conveter_aten__fused_dropout_default(
    self: Tensor,
    p: float,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten._fused_dropout.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._fused_dropout.out)
def conveter_aten__fused_dropout_out(
    self: Tensor,
    p: float,
    generator: Optional[Generator] = None,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_fused_dropout.out(Tensor self, float p, Generator? generator=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten._fused_dropout.out ge_converter is not implemented!")
