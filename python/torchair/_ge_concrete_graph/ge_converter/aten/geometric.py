from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.geometric.default)
def conveter_aten_geometric_default(
    self: Tensor,
    p: float,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::geometric(Tensor self, float p, *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.geometric.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.geometric.out)
def conveter_aten_geometric_out(
    self: Tensor,
    p: float,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::geometric.out(Tensor self, float p, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.geometric.out ge_converter is not implemented!")
