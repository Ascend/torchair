from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.bernoulli_.Tensor)
def conveter_aten_bernoulli__Tensor(
    self: Tensor,
    p: Tensor,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bernoulli_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bernoulli_.float)
def conveter_aten_bernoulli__float(
    self: Tensor,
    p: float = 0.5,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bernoulli_.float ge_converter is not implemented!")
